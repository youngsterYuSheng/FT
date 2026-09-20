# TDSign

Official PyTorch implementation of **TDSign** for continuous sign language recognition (CSLR).

TDSign focuses on fine-grained temporal details that can be overlooked within holistic sign motion. It contains two main components:

* **Adaptive Temporal Detail Enhancement (ATDE)** extracts multiscale temporal residuals and selectively adjusts local details according to their reliability and holistic motion context.
* **Detail-Aware Spatiotemporal Supervision (DSTS)** preserves enhanced spatial details and models their temporal variations to provide auxiliary CTC supervision during training. DSTS is removed at inference.

## Framework

<p align="center">
  <img src="figures/framework.png" width="95%">
</p>

## Main Results

Word error rate (WER, %) on three CSLR benchmarks:

| Method | PHOENIX14 Dev | PHOENIX14 Test | PHOENIX14-T Dev | PHOENIX14-T Test | CSL-Daily Dev | CSL-Daily Test |
| ------ | ------------: | -------------: | --------------: | ---------------: | ------------: | -------------: |
| TDSign |     `<value>` |      `<value>` |       `<value>` |        `<value>` |     `<value>` |      `<value>` |

Please replace the entries above with the final results reported in the paper.

## Requirements

```bash
conda create -n tdsign python=3.8
conda activate tdsign
pip install -r requirements.txt
```

The main dependencies include:

* Python 3.8
* PyTorch
* torchvision
* NumPy
* OpenCV
* ctcdecode

Exact package versions are provided in `requirements.txt`.

## Datasets

Our experiments are conducted on:

* PHOENIX-2014
* PHOENIX-2014T
* CSL-Daily

Please download the datasets from their official sources and organize them according to the preprocessing instructions of this repository.

The dataset path can be specified in the corresponding configuration file:

```yaml
dataset_root: /path/to/dataset
```

## Training

Train TDSign using:

```bash
python <training_script>.py \
    --config <path_to_config>
```

ATDE is inserted into the visual backbone, while DSTS provides auxiliary supervision only during training.

## Evaluation

Evaluate a trained model using:

```bash
python <evaluation_script>.py \
    --config <path_to_config> \
    --weights <path_to_checkpoint>
```

DSTS is automatically removed during inference and therefore introduces no additional inference overhead.

## Pretrained Models

| Dataset       | Checkpoint                      |   Dev WER |  Test WER |
| ------------- | ------------------------------- | --------: | --------: |
| PHOENIX-2014  | [Download](`<checkpoint_link>`) | `<value>` | `<value>` |
| PHOENIX-2014T | [Download](`<checkpoint_link>`) | `<value>` | `<value>` |
| CSL-Daily     | [Download](`<checkpoint_link>`) | `<value>` | `<value>` |

## Key Implementation

The main components are implemented in:

* `modules/resnet.py`: visual backbone and ATDE
* `slr_network.py`: DSTS and the overall recognition network

The ATDE ablation setting can be selected through `type`:

| `type` | Learned Gain | Reliability | Description                         |
| -----: | :----------: | :---------: | ----------------------------------- |
|    `0` |       —      |      —      | Equal-weight residual aggregation   |
|    `1` |       ✓      |      —      | Gain estimation without reliability |
|    `2` |       ✓      |      ✓      | Full ATDE                           |
|    `3` |       —      |      ✓      | Direct reliability weighting        |

The complete model uses `type=2`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tdsign,
  title     = {<Paper Title>},
  author    = {<Authors>},
  booktitle = {<Conference>},
  year      = {<Year>}
}
```

The complete citation will be updated after publication.

## Acknowledgements

We thank the authors of the public CSLR datasets and open-source implementations that supported this work.

## License

This project is released under the `<license name>` License. See `LICENSE` for details.
