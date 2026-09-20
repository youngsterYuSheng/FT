import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class DWA(nn.Module):
    def __init__(self, in_channels):
        super(DWA, self).__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        gap_out = x.mean(dim=[-2, -1])

        # Normalize spatial attention separately for each frame.
        x_reshaped = x.transpose(1, 2).reshape(B * T, C, H, W)
        attn_logits = self.attn_conv(x_reshaped)

        B_T, _, H_, W_ = attn_logits.shape
        attn_weights = torch.softmax(attn_logits.view(B_T, 1, -1), dim=-1).view(B_T, 1, H_, W_)

        x_attended = (x_reshaped * attn_weights).sum(dim=[-2, -1])
        attn_out = x_attended.view(B, T, C).transpose(1, 2)
        return gap_out + attn_out


class TDM(nn.Module):
    def __init__(self, hidden_size):
        super(TDM, self).__init__()
        self.dw_k3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size, bias=False)
        self.dw_k5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size, bias=False)

        self.shrink_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, stride=1, padding=0, groups=hidden_size,
                                     bias=False)

        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.act = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=False)

    def forward(self, x):
        x_multi = x + self.dw_k3(x) + self.dw_k5(x)

        x_shrink = self.shrink_conv(x_multi)

        out = self.pointwise(x_shrink)
        out = self.bn(out)
        out = self.act(out)

        return self.pool(out)


class DSTS(nn.Module):
    def __init__(self, in_channels_list=[128, 256, 512], hidden_size=1024):
        super(DSTS, self).__init__()

        self.spatial_poolers = nn.ModuleList([
            DWA(c) for c in in_channels_list
        ])

        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c, hidden_size, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])

        self.temporal_downsample = nn.Sequential(
            TDM(hidden_size),
            TDM(hidden_size)
        )

    def forward(self, res_list):
        multi = []
        for i, res_feat in enumerate(res_list):
            x = self.spatial_poolers[i](res_feat)
            x = self.projectors[i](x)
            x = self.temporal_downsample(x)
            multi.append(x)
        return multi


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)()
        
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        self.light_ds = DSTS(in_channels_list=[128, 256, 512], hidden_size=hidden_size)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        vis_data = {}
        if len(x.shape) == 5:
            feat_l4, feat_l3,res = self.conv2d(x.permute(0, 2, 1, 3, 4))
            framewise = feat_l4.mean(dim=[-2, -1])
            framewise = framewise.permute(0, 2, 1)

            framewise = framewise.permute(0, 2, 1)

        else:
            framewise = x

        res2 = []

        if self.training and len(x.shape) == 5:
            ds_features = self.light_ds([res[1], res[2], res[3]])
            for i in range(3):
                feat = ds_features[i].permute(2, 0, 1)
                logits = self.classifier(feat)
                res2.append(logits)

        conv1d_outputs = self.conv1d(framewise, len_x)
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return_dict = {
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "res2": res2,
        }
        if vis_data:
            return_dict["vis_data"] = vis_data

        if 'loss_LiftPool_u' in conv1d_outputs:
            return_dict["loss_LiftPool_u"] = conv1d_outputs.get('loss_LiftPool_u', 0)
            return_dict["loss_LiftPool_p"] = conv1d_outputs.get('loss_LiftPool_p', 0)

        return return_dict

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['ConvCTC']
            elif k == 'SeqCTC':
                total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['SeqCTC']
                if len(ret_dict["res2"]) > 0:
                    for i in range(3):
                        loss += self.loss['CTCLoss'](ret_dict["res2"][i].log_softmax(-1),
                                                     label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                     label_lgt.cpu().int()).mean()

            elif k == 'Dist':
                total_loss['Dist'] = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
                loss += total_loss['Dist']
            elif k == 'Cu':
                total_loss['Cu'] = weight * ret_dict["loss_LiftPool_u"]
                loss += total_loss['Cu']
            elif k == 'Cp':
                total_loss['Cp'] = weight * ret_dict["loss_LiftPool_p"]
                loss += total_loss['Cp']
        return loss, total_loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['mse_loss'] = nn.MSELoss()
        return self.loss