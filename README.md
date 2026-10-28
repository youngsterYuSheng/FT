# TDSign

Official PyTorch implementation of the manuscript:

> **TDSign: Adaptive Enhancement of Fine-Grained Temporal Details for Continuous Sign Language Recognition**  
> Weihuang Yu, Guangqian Kong, and Xun Duan  
> College of Computer Science and Technology, Guizhou University

TDSign captures fine-grained temporal details that can be overlooked within holistic sign motion. It contains two complementary components:




##

在三个CSLR基准测试中的词错误率（WER，%）。数值越低越好。TDSign仅使用RGB输入。

|方法|PHOENIX14开发集|PHOENIX14测试集|PHOENIX14-T开发集|PHOENIX14-T测试集|CSL-Daily开发集|CSL-Daily测试集|

| TDSign | ** | **16.9** | **16.4** | **17.8** | **25.0** | **24.1** |

##消融研究
在PHOENIX14数据集上开展消融实验，以验证ATDE和DSTS的有效性。报告词错误率（WER，%），数值越低越好。

|设置|PHOENIX14开发集|PHOENIX14测试集|
|---|---:|---:|
|基线| 18.9 | 18.9 |
|仅使用ATDE| 17.4 | 17.6 |
|仅使用DSTS| 18.2 | 18.4 |
|带 ATDE + DSTS（TDSign）| **16.8** | **16.9** |

##先决条件

- Python 3.8
- PyTorch
- `ctcdecode`用于束搜索解码
- `sclite`（可选）用于生成详细评估统计信息

使用以下命令安装其余依赖项：

```bash
pip install -r requirements.txt
```

##实现

主要组件位于：

- `modules/resnet.py`: ResNet34 visual backbone and ATDE;
- `slr_network.py`: DSTS, sequence modeling, and auxiliary CTC supervision.

ATDE is inserted after Stages 2, 3, and 4 of the visual backbone. DSTS contains Detail-aware Weighted Aggregation (DWA) and Temporal Detail Modeling (TDM), shares its temporal module across stages, and reuses the classifier of the main recognition path.

The full model uses `type=2`. The other settings are provided for reproducing the ATDE gain ablation:

| `type` | Learned Gain | Reliability | Setting |
|---:|:---:|:---:|---|
| `0` | - | - | Equal-weight residual aggregation |
| `1` | Yes | - | Gain estimation without reliability cues |
| `2` | Yes | Yes | Full ATDE |
| `3` | - | Yes | Direct reliability weighting |

## Data Preparation

TDSign is evaluated on:

- [RWTH-PHOENIX-Weather 2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
- [RWTH-PHOENIX-Weather 2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/)

Please request or download each dataset from its official provider and follow its license. The datasets are not redistributed in this repository.

The expected directory structure is:

```text
dataset/
├── phoenix2014/
│   └── phoenix-2014-multisigner/
│       └── features/
│           └── fullFrame-256x256px/
├── phoenix2014-T/
│   └── features/
│       └── fullFrame-256x256px/
└── CSL-Daily/
    └── sentence/
        └── frames_256x256/
```

Symbolic links can be used instead of copying the datasets:

```bash
mkdir -p dataset/phoenix2014
ln -s /path/to/phoenix-2014-multisigner dataset/phoenix2014/phoenix-2014-multisigner
ln -s /path/to/PHOENIX-2014-T dataset/phoenix2014-T
ln -s /path/to/CSL-Daily dataset/CSL-Daily
```

Generate the resized frames and annotation metadata with the preprocessing scripts supplied in `preprocess/`:

```bash
cd preprocess

# PHOENIX-2014
python dataset_preprocess.py --process-image --multiprocessing

# PHOENIX-2014T
python dataset_preprocess-T.py --process-image --multiprocessing

# CSL-Daily
python dataset_preprocess-CSL-Daily.py --process-image --multiprocessing
```

## Training

Select the target dataset in the configuration or pass it from the command line. For example:

```bash
# PHOENIX-2014
python main.py --device 0 --dataset phoenix2014 \
  --work-dir ./work_dir/phoenix2014/

# PHOENIX-2014T
python main.py --device 0 --dataset phoenix2014-T \
  --work-dir ./work_dir/phoenix2014-T/

# CSL-Daily
python main.py --device 0 --dataset CSL-Daily \
  --work-dir ./work_dir/CSL-Daily/
```

The experiments reported in the paper use:

- an ImageNet-pretrained ResNet34 backbone;
- the TLP temporal encoder followed by a two-layer BiLSTM with hidden dimension 1,024;
- Adam with an initial learning rate of `1e-4` and weight decay of `1e-4`;
- a batch size of 2 and 80 training epochs;
- learning-rate decay by a factor of 0.2 at epochs 40 and 60;
- `224 x 224` RGB crops;
- one NVIDIA L20 GPU.

## Evaluation

To evaluate a model trained with this repository, run:

```bash
python main.py --device 0 --dataset phoenix2014 --phase test \
  --load-weights /path/to/trained_model.pt \
  --work-dir ./work_dir/phoenix2014_test/
```

Replace `phoenix2014` with `phoenix2014-T` or `CSL-Daily` for the other datasets.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{yu2026tdsign,
  title  = {TDSign: Adaptive Enhancement of Fine-Grained Temporal Details for Continuous Sign Language Recognition},
  author = {Yu, Weihuang and Kong, Guangqian and Duan, Xun},
  year   = {2026}
}
```

The citation will be updated after publication.

## Acknowledgments

This codebase builds upon [VAC](https://github.com/VIPL-SLP/VAC_CSLR), [CorrNet](https://github.com/hulianyuyy/CorrNet), and [SlowFastSign](https://github.com/kaistmm/SlowFastSign). We thank the authors for releasing their code.

This work was supported by the Guizhou Provincial Basic Research Program (Natural Science) under Grant No. Qiankehe Foundation MS[2026]081.
