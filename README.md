# LVS-Net

A lightweight retinal vessel segmentation network (0.71 M parameters,
2.74 MB) with focal modulation attention and feature refinement.

Trained and evaluated on DRIVE, STARE and CHASE_DB.

## Contents

    lvs_net.py          the model
    common.py           data loading, metrics, thresholding
    train.py            stage 1: train from scratch
    finetune.py         stage 2: continue with a cosine-decayed learning rate
    evaluate.py         per-image metrics for a checkpoint on a test set
    select_threshold.py evaluation with a threshold frozen on validation data
    benchmark.py        inference latency, throughput and memory
    predict.py          run a checkpoint on a folder of images
    tta.py              evaluation with test-time augmentation
    checkpoints/        trained weights for each dataset
    requirements.txt

Data is distributed separately (see **Data** below) because of its size.

## Install

    pip install -r requirements.txt

## Run a trained model

    python evaluate.py --weights checkpoints/LVS-Net_STARE.pt \
                       --data data/STARE/Test

    python predict.py --weights checkpoints/LVS-Net_CHASE.pt \
                      --images data/CHASE/Test/Images \
                      --masks  data/CHASE/Test/GT \
                      --out    predictions/CHASE

`--masks` is optional; without it `predict.py` only writes probability maps.

## Train from scratch

    python train.py --data data/CHASE --epochs 60 --out runs/chase

then fine-tune with a decaying learning rate:

    python finetune.py --data data/CHASE --init runs/chase/best.pt \
                       --epochs 150 --out runs/chase_finetuned

Expected layout:

    data/<DATASET>/Train/Images   data/<DATASET>/Train/GT
    data/<DATASET>/Test/Images    data/<DATASET>/Test/GT

Image and mask filenames must sort into the same order.

## Results

Every test image is evaluated. The binarisation threshold is chosen once on
the held-out validation images and then frozen, so no test annotation is
used in setting it. These are the numbers reported in the paper.

| dataset | Dice | Jaccard | Sn | Sp | Acc | threshold |
|---------|------|---------|----|----|-----|-----------|
| DRIVE (20 test images)   | 83.74 | 72.04 | 87.02 | 98.01 | 97.05 | 0.370 |
| STARE (4 test images)    | 83.04 | 71.17 | 88.48 | 97.99 | 97.27 | 0.490 |
| CHASE_DB (8 test images) | 83.51 | 71.72 | 82.27 | 98.62 | 97.21 | 0.490 |

Reproduce with `select_threshold.py`, which also prints two weaker protocols
for comparison: a threshold optimised per test image against that image's own
mask, and the single best threshold on the test set. Both read the test
ground truth, so neither is a valid headline number. Freezing the threshold
costs only 0.16-0.69 Dice relative to per-image tuning, which is the useful
point: the model is not sensitive to where the threshold sits.

| dataset | frozen | per-image (uses GT) | best-on-test (uses GT) |
|---------|--------|---------------------|------------------------|
| DRIVE    | 83.74 | 84.11 | 83.98 |
| STARE    | 83.04 | 83.73 | 83.40 |
| CHASE_DB | 83.51 | 83.67 | 83.58 |

Per image, Jaccard and Dice satisfy `IoU = Dice / (2 - Dice)` exactly. After
averaging over images the mean Jaccard sits very slightly *above* the value
implied by the mean Dice (by 0.0002 to 0.0015 here), which is what Jensen's
inequality requires, since that mapping is convex. A reported Jaccard below
the implied value is not reachable and indicates an error.

## Training settings

Adam, learning rate 1e-3, soft Dice loss, batch size 8, images resized to
512x512. 90% of the training images are used for training and 10% held out
for validation; the epoch with the best validation Dice is kept. Test images
are never used for training or for choosing the saved epoch.

Datasets were trained for 60 epochs and then fine-tuned for a
further 150 epochs with the learning rate decayed from 5e-4 to 0 on a cosine
schedule (`finetune.py`), with random flips and 90-degree rotations applied
on the fly on top of the offline augmentation. Both were trained on an
NVIDIA GB10.

## Data

| dataset | train | test | augmented train set |
|---------|-------|------|---------------------|
| DRIVE    | 20 images (21-40) | 20 images (01-20) | 880 |
| STARE    | 16 images | 4 images (im0044, im0077, im0081, im0082) | 864 |
| CHASE_DB | 20 images | 8 images | 720 |

The DRIVE training and test sets are the official split: the 880 augmented
training images are built from photographs 21-40 only, and photographs 01-20
are used for testing and never seen during training.

Training sets are augmented with rotations and flips; the
DRIVE training set additionally uses photometric augmentation (gamma,
blurring, sharpening, Gaussian noise, saturation, histogram equalisation).

**STARE uses the first observer's (A. Hoover, `ah`) annotations for both
training and testing.** The two STARE observers trace vessels at noticeably
different thicknesses -- about 7.6% versus 10.9% of pixels -- so mixing them
between training and testing distorts results.

## Inference cost

Measured with `benchmark.py` at 512x512, batch size 1, median of 30 timed
runs after 8 warm-up runs:

| device | ms/image | FPS | peak memory |
|--------|----------|-----|-------------|
| NVIDIA GB10 (GPU)  |  45.0 | 22.2 | 228.3 MB |
| CPU, 20 threads    | 192.1 |  5.2 | -- |
| CPU, single thread | 395.2 |  2.5 | -- |

## Notes on evaluation

Metrics are computed over all test images. Three pitfalls are worth naming,
because each inflates scores:

* evaluating a single image rather than the whole test set;
* tuning a separate threshold per pixel row of one image, which uses the
  ground truth far more finely than a per-image threshold does;
* choosing the threshold on the test set at all, including a per-image
  F1-optimal threshold, since it reads the test mask.

`select_threshold.py` avoids all three. `evaluate.py` reports the per-image
protocol, which is common in the literature and is included for comparison
with published numbers, but the frozen-threshold result is the one to quote.

Test-time augmentation is available through `tta.py`, which reports plain,
4-way and 8-way TTA, each with per-image F1-optimal thresholds and with a
single threshold chosen on the held-out validation images. The numbers in
the table above are **without** TTA.
