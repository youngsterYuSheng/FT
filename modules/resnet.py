
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ATDE', 'ResNet', 'resnet18', 'resnet34']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ATDE(nn.Module):
    def __init__(self, in_channels, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.holistic_in = nn.Conv3d(self.in_channels, self.inter_channels, 1, bias=False)
        self.detail_in = nn.Conv3d(self.in_channels, self.inter_channels, 1, bias=False)

        self.holistic_path = nn.Sequential(
            nn.Conv3d(self.inter_channels, self.inter_channels, (3, 3, 3), padding=1, groups=self.inter_channels,
                      bias=False),
            nn.GroupNorm(groups, self.inter_channels),
            nn.SiLU(),
            nn.Conv3d(self.inter_channels, self.inter_channels, (3, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1),
                      groups=self.inter_channels, bias=False),
            nn.GroupNorm(groups, self.inter_channels)
        )

        self.detail_op = nn.Sequential(
            nn.Conv3d(self.inter_channels, self.inter_channels, 1, bias=False),
            nn.GroupNorm(groups, self.inter_channels),
            nn.SiLU()
        )
        self.detail_gain_op = nn.Sequential(
            nn.Conv3d(self.in_channels + 4, self.inter_channels, 1, bias=False),
            nn.GroupNorm(groups, self.inter_channels),
            nn.SiLU(),
            nn.Conv3d(self.inter_channels, self.inter_channels, 1, bias=True),
            nn.Tanh()
        )

        self.dgc_conv = nn.Conv1d(self.in_channels, self.in_channels // 16, 1)
        self.dgc_fc = nn.Sequential(
            nn.LayerNorm(self.in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // 16, 1),
            nn.Tanh()
        )
        self.fusion_gate = nn.Conv3d(self.in_channels, self.inter_channels, 1, bias=True)
        self.fusion_op = nn.Conv3d(self.inter_channels * 3, self.in_channels, 1, bias=False)

        self.res_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.base_diff_scale = nn.Parameter(torch.ones(1) * 2.0)

        self._init_fdr_weights()

    def _init_fdr_weights(self):
        with torch.no_grad():
            self.holistic_in.weight.zero_()
            self.detail_in.weight.zero_()
            eye = torch.eye(self.inter_channels, device=self.holistic_in.weight.device)
            self.holistic_in.weight[:, :self.inter_channels, 0, 0, 0].copy_(eye)
            self.detail_in.weight[:, self.inter_channels:, 0, 0, 0].copy_(eye)

            nn.init.zeros_(self.detail_gain_op[-2].weight)
            nn.init.zeros_(self.detail_gain_op[-2].bias)
            nn.init.zeros_(self.dgc_fc[2].weight)
            nn.init.zeros_(self.dgc_fc[2].bias)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)

            self.fusion_op.weight.zero_()
            self.fusion_op.weight[:self.inter_channels, :self.inter_channels, 0, 0, 0].copy_(eye)
            self.fusion_op.weight[self.inter_channels:, self.inter_channels:2 * self.inter_channels,
                                  0, 0, 0].copy_(eye)

    @staticmethod
    def _temporal_avg(x, kernel_size):
        pad = kernel_size // 2
        x = F.pad(x, (0, 0, 0, 0, pad, pad), mode='replicate')
        return F.avg_pool3d(x, (kernel_size, 1, 1), stride=1, padding=0)

    def detail_gain(self, detail_list, h_out):
        eps = 1e-6
        h_out_32 = h_out.float()
        pool_h = torch.mean(h_out_32.abs(), dim=(3, 4))
        d_amplified, gain_list, reliability_list = [], [], []

        for detail in detail_list:
            detail = detail.float()
            energy = torch.mean(detail.abs(), dim=1, keepdim=True)

            temporal_ref = self._temporal_avg(detail, 3)
            spatial_ref = F.avg_pool3d(detail, (1, 3, 3), stride=1, padding=(0, 1, 1))

            temporal_error = torch.mean((detail - temporal_ref).abs(), dim=1, keepdim=True)
            spatial_error = torch.mean((detail - spatial_ref).abs(), dim=1, keepdim=True)
            temporal_coh = energy / (energy + temporal_error + eps)
            spatial_coh = energy / (energy + spatial_error + eps)

            noise_floor = torch.mean(
                (detail - 0.5 * (temporal_ref + spatial_ref)).abs(), dim=1, keepdim=True)
            snr = energy / (energy + noise_floor + eps)

            compatibility = F.cosine_similarity(detail, h_out_32, dim=1, eps=eps).unsqueeze(1)
            compatibility = 0.5 * (compatibility + 1.0)

            reliability = torch.clamp(
                (temporal_coh + spatial_coh + snr + compatibility) * 0.25,
                min=0.0, max=1.0)

            gain_input = torch.cat(
                [h_out_32, detail, temporal_coh, spatial_coh, snr, compatibility], dim=1)
            gain_map = self.detail_gain_op(gain_input)

            pool_d = torch.mean(detail.abs(), dim=(3, 4)) * torch.mean(
                reliability, dim=(3, 4))
            energy_t = torch.cat([pool_h, pool_d], dim=1)
            gain_t = self.dgc_fc(self.dgc_conv(energy_t).transpose(1, 2))

            gain_map = torch.tanh(
                gain_map + gain_t.view(gain_t.size(0), 1, gain_t.size(1), 1, 1))
            safe_base_scale = torch.clamp(self.base_diff_scale.abs(), max=2.0)
            dynamic_scale = torch.clamp(
                1.0 + 0.5 * safe_base_scale * gain_map,
                min=0.0, max=2.0)

            d_amplified.append(detail * dynamic_scale)
            gain_list.append(gain_t)
            reliability_list.append(reliability)

        d_sig = torch.stack(d_amplified, dim=0).sum(dim=0)
        band_gain_t = torch.stack(gain_list, dim=1)
        gain_t = torch.mean(band_gain_t, dim=1)
        detail_reliability = torch.stack(reliability_list, dim=1)
        return d_sig, gain_t, band_gain_t, detail_reliability

    def forward(self, x):
        B, C, T, H, W = x.shape
        orig_type = x.dtype

        c_h = self.holistic_in(x)
        c_d = self.detail_in(x)

        h_out = self.holistic_path(c_h)

        with torch.cuda.amp.autocast(enabled=False):
            c_d_32 = c_d.float()

            t_avg = self._temporal_avg(c_d_32, 3)
            t_avg_5 = self._temporal_avg(c_d_32, 5)
            t_avg_7 = self._temporal_avg(c_d_32, 7)
            t_res = c_d_32 - t_avg
            t_res_5 = t_avg - t_avg_5
            t_res_7 = t_avg_5 - t_avg_7

            detail_list = [t_res, t_res_5, t_res_7]

            d_sig, gain_t, band_gain_t, detail_reliability = self.detail_gain(
                detail_list, h_out)
            d_sig = torch.clamp(d_sig, min=-50.0, max=50.0).to(orig_type)

        d_out = self.detail_op(d_sig)

        fusion_weight = 1.0 + torch.tanh(
            self.fusion_gate(torch.cat([h_out, d_out], dim=1)))
        d_out = d_out * fusion_weight
        fused_out = self.fusion_op(torch.cat([h_out, d_out, h_out * d_out], dim=1))

        refined_feat = x + self.res_scale * fused_out

        vis = {
            "h_out": h_out.detach(),
            "d_out": d_out.detach(),
            "curve_gain_t": gain_t.detach(),
            "band_gain_t": band_gain_t.detach(),
            "detail_reliability": detail_reliability.detach(),
            "fusion_weight": fusion_weight.detach(),
            "val_res_scale": self.res_scale.detach(),
            "val_diff_scale": self.base_diff_scale.detach(),
            "refined_feat": refined_feat.detach(),
            "fused_out": fused_out.detach(),
        }

        return refined_feat, vis


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.fdr_l2 = ATDE(in_channels=128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.fdr_l3 = ATDE(in_channels=256)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fdr_l4 = ATDE(in_channels=512)
        self.vis_data_cache = {}

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.vis_data_cache.clear()
        N, C, T, H, W = x.size()
        res = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        res.append(x)
        x = self.layer2(x)
        x, fdr_vis = self.fdr_l2(x)
        self.vis_data_cache['stage2_sts'] = fdr_vis
        res.append(x)
        x = self.layer3(x)
        x, fdr_vis = self.fdr_l3(x)
        self.vis_data_cache['stage3_sts'] = fdr_vis
        res.append(x)
        feat_l3 = x
        x = self.layer4(x)
        x, fdr_vis = self.fdr_l4(x)
        self.vis_data_cache['stage4_sts'] = fdr_vis
        res.append(x)
        return x, feat_l3, res


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'], map_location=torch.device('cpu'))
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    import gc
    gc.collect()
    return model


def _load_pretrained_weights(model, url):
    checkpoint = model_zoo.load_url(url, map_location=torch.device('cpu'))
    state_dict = model.state_dict()
    for k, v in checkpoint.items():
        if 'conv' in k or 'downsample.0.weight' in k:
            if v.dim() == 4:
                v = v.unsqueeze(2)
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k] = v
    model.load_state_dict(state_dict, strict=False)
    print(f"Pretrained weights from {url.split('/')[-1]} loaded successfully.")
    del checkpoint
    import gc
    gc.collect()
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model = _load_pretrained_weights(model, model_urls['resnet34'])
    return model

