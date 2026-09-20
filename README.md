TDSign

Official PyTorch implementation of TDSign: Adaptive Enhancement of Fine-Grained Temporal Details for Continuous Sign Language Recognition.

TDSign is designed to capture fine-grained temporal details that can be overlooked within holistic sign motion. It contains two complementary components:

Adaptive Temporal Detail Enhancement (ATDE) extracts multiscale temporal residuals and combines detail reliability with holistic motion context to selectively adjust informative local details.

Detail-Aware Spatiotemporal Supervision (DSTS) preserves enhanced spatial details and models their variations across temporal scales to provide auxiliary CTC supervision. DSTS is used only during training and removed at inference.

Method

TDSign follows a CNN-BiLSTM CSLR pipeline. A ResNet34 visual backbone extracts frame-wise features, a 1D temporal encoder models local temporal patterns, and a two-layer BiLSTM captures long-range dependencies.

ATDE is inserted after Stages 2, 3, and 4 of the visual backbone. DSTS adapts the corresponding intermediate features into auxiliary sequences using:

Detail-aware Weighted Aggregation (DWA) for spatial compression while preserving localized detail responses;

Temporal Detail Modeling (TDM) for lightweight multiscale temporal modeling.

Results

Word error rate (WER, %) on three continuous sign language recognition benchmarks. Lower is better.

Method

PHOENIX14 Dev

PHOENIX14 Test

PHOENIX14-T Dev

PHOENIX14-T Test

CSL-Daily Dev

CSL-Daily Test

TDSign

16.8

16.9

16.4

17.8

25.0

24.1

TDSign uses RGB input only.

Datasets

Experiments are conducted on three public CSLR datasets:

PHOENIX-2014: 6,841 German Sign Language videos from television weather forecasts, with 1,295 glosses.

PHOENIX-2014T: an extension of PHOENIX-2014 with German translation annotations and a vocabulary of 1,085 glosses.

CSL-Daily: 20,654 Chinese Sign Language videos covering daily-life topics, with 2,000 glosses.

Please obtain the datasets from their official providers and follow their licenses and terms of use. Dataset files are not distributed with this repository.

Implementation Details

The experiments reported in the paper use the following settings:

ImageNet-pretrained ResNet34 visual backbone;

temporal encoder following the TLP configuration;

two-layer BiLSTM with a hidden dimension of 1,024;

Adam optimizer;

80 training epochs;

batch size of 2;

initial learning rate of $1\times10^{-4}$;

weight decay of $1\times10^{-4}$;

learning-rate decay by a factor of 0.2 at epochs 40 and 60;

$224\times224$ RGB crops as input;

training on a single NVIDIA L20 GPU.

Installation

Create a Python environment and install the dependencies required by the project:

conda create -n tdsign python=3.8
conda activate tdsign
pip install -r requirements.txt

The project requires PyTorch, NumPy, OpenCV, and a CTC decoder compatible with the supplied training framework.

Code Structure

The main TDSign components are implemented in:

modules/resnet.py: ResNet34 visual backbone and ATDE;

slr_network.py: DSTS, the recognition network, and auxiliary CTC supervision.

DSTS is active only during training and introduces no additional inference overhead.

ATDE Ablation Settings

The ATDE implementation provides the following experimental settings through type:

type

Learned Gain

Reliability

Description

0

-

-

Equal-weight residual aggregation

1

Yes

-

Learned gain estimation without reliability cues

2

Yes

Yes

Full ATDE used by TDSign

3

-

Yes

Direct reliability weighting without learned gains

The complete TDSign model uses type=2.

Ablation Results

ATDE and DSTS

ATDE

DSTS

Dev WER

Test WER

-

-

18.9

18.9

Yes

-

17.5

17.6

-

Yes

18.0

18.3

Yes

Yes

16.8

16.9

Reliability-Guided Gain Estimation

Learned Gain

Reliability

Dev WER

Test WER

-

-

17.4

17.6

Yes

-

17.2

17.7

-

Yes

17.1

17.4

Yes

Yes

16.8

16.9

These results show that combining reliability cues with learned gain estimation gives the best performance among the evaluated variants.

Citation

If this work is useful for your research, please cite:

@misc{yu2026tdsign,
  title  = {TDSign: Adaptive Enhancement of Fine-Grained Temporal Details for Continuous Sign Language Recognition},
  author = {Yu, Weihuang and Kong, Guangqian and Duan, Xun},
  year   = {2026}
}

The citation information will be updated after publication.

Acknowledgment

This work was supported by the Guizhou Provincial Basic Research Program (Natural Science) under Grant No. Qiankehe Foundation MS[2026]081.

