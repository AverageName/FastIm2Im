## Denoising block

This folder consists of implementation of 2 models: 
- DnCNN (based on https://github.com/cszn/KAIR)
- ASAPNet generator (based on https://github.com/tamarott/ASAPNet). Disclaimer: pixelwise network, performs poorly

## Quickstart

1. `pip install -r requirement.txt`
2. prepare data (used BSD500 from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/; for preprocessing steps follow `data/bsd_preprocess.py`):
  - put GT (ground truth) train images to `trainsets/bsdH`
  - put noisy train images to `trainsets/bsdL`
  - put GT (ground truth) test images to `testsets/bsdH`
  - put noisy test images to `testsets/bsdL`
 
3. Run training scripts: `train_dncnn.py`, `train_asapnet.py`
  - Training configs are stored in `options/train_dncnn.json` and `options/train_asap.json` respectively
  - Training procedures includes testing on the test set, scores (SSIM, PSNR) at test are calculated once per 800 iterations by default 

4. Use testing scripts `test_dncnn.py`, `test_asapnet.py` if you have holdout test data not included while training. 
  - Easier to put all the test data to `testsets/bsd[LH]` and look for the metrics' dynamics


## Scores 

### PSNR (dB)

| Models        | DnCNN | ASAPNet |
|---------------|-------|---------|
| $\sigma$ = 15 | 26.31 | 23.67   |
| $\sigma$ = 25 | 24.65 | 21.73   |
| $\sigma$ = 50 | 21.88 | 17.64   |

### SSIM

| Models        | DnCNN | ASAPNet |
|---------------|-------|---------|
| $\sigma$ = 15 | 0.86  | 0.77    |
| $\sigma$ = 25 | 0.78  | 0.61    |
| $\sigma$ = 50 | 0.65  | 0.42    |

### Inference Time

| Models        | DnCNN  | ASAPNet |
|---------------|--------|---------|
| $\sigma$ = 15 | 0.056s | 0.012s  |
| $\sigma$ = 25 | 0.059s | 0.011s  |
| $\sigma$ = 50 | 0.057s | 0.012s  |
