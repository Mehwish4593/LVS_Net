"""
select_threshold.py -- evaluate with a threshold frozen on validation data.

"""

import argparse
import os

import numpy as np

from common import (SIZE, best_global_threshold, f1_threshold, load_images,
                    load_masks, load_model, mean_metrics, metrics)

KEYS = ("dice", "jaccard", "sensitivity", "specificity", "accuracy")


def validation_split(data_dir, how, seed=42):
    """Indices held out for validation, reproducing the split training used.

    The STARE and CHASE checkpoints were trained with PyTorch's
    random_split(generator=manual_seed(42)) holding out len//10; the DRIVE
    checkpoint was trained in Keras with sklearn's train_test_split(
    test_size=0.20, random_state=42). The two use different random number
    generators, so each must be reproduced with its own.
    """
    names = sorted(os.listdir(os.path.join(data_dir, "Train", "Images")))
    n = len(names)

    if how == "torch":
        import torch
        n_val = max(1, n // 10)
        _, va = torch.utils.data.random_split(
            range(n), [n - n_val, n_val],
            generator=torch.Generator().manual_seed(seed))
        return sorted(va.indices), names

    from sklearn.model_selection import train_test_split
    _, val_idx = train_test_split(np.arange(n), test_size=0.20, random_state=seed)
    return sorted(val_idx), names


def load_subset(data_dir, idx, names):
    X = np.zeros((len(idx), SIZE, SIZE, 3), np.float32)
    Y = np.zeros((len(idx), SIZE, SIZE), np.float32)
    from PIL import Image
    img_dir = os.path.join(data_dir, "Train", "Images")
    gt_dir = os.path.join(data_dir, "Train", "GT")
    gts = sorted(os.listdir(gt_dir))
    for k, i in enumerate(idx):
        X[k] = np.asarray(Image.open(os.path.join(img_dir, names[i])).convert("RGB")
                          .resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.
        Y[k] = np.asarray(Image.open(os.path.join(gt_dir, gts[i])).convert("L")
                          .resize((SIZE, SIZE), Image.NEAREST), np.float32) / 255. > 0.5
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="folder holding Train/ and Test/")
    ap.add_argument("--split", choices=["torch", "sklearn"], default=None,
                    help="how training split off validation (default: see below)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

       # The DRIVE weights came from the original Keras run, which split off
    # validation with sklearn; STARE and CHASE were trained in PyTorch, which
    # splits differently. Each checkpoint's threshold must be chosen on the
    # same validation images its own training held out.
    how = args.split or ("sklearn" if "DRIVE" in os.path.basename(args.weights)
                         else "torch")
    predict = load_model(args.weights)

    idx, names = validation_split(args.data, how)
    Xva, Yva = load_subset(args.data, idx, names)
    Pva = predict(Xva)
    t_frozen, val_dice = best_global_threshold(Pva, Yva)
    print("validation: %d of %d training images (%s split), frozen threshold %.3f (val Dice %.2f)"
          % (len(idx), len(names), how, t_frozen, 100 * val_dice))

    Xte, te_names = load_images(os.path.join(args.data, "Test", "Images"))
    Yte = load_masks(os.path.join(args.data, "Test", "GT"))
    P = predict(Xte)

    frozen = mean_metrics([metrics((P[i] >= t_frozen).astype(np.float32), Yte[i])
                           for i in range(len(Yte))])
    per_img = mean_metrics([metrics((P[i] >= f1_threshold(P[i], Yte[i])).astype(np.float32), Yte[i])
                            for i in range(len(Yte))])
    oracle_t, _ = best_global_threshold(P, Yte)
    oracle = mean_metrics([metrics((P[i] >= oracle_t).astype(np.float32), Yte[i])
                           for i in range(len(Yte))])

    print("\n%s%-34s %6s %6s %6s %6s %6s" % (
        args.label + "\n" if args.label else "",
        "protocol", "Dice", "Jacc", "Sn", "Sp", "Acc"))
    print("-" * 70)
    for lbl, r in [("frozen on validation (thr %.3f)" % t_frozen, frozen),
                   ("per-image F1-optimal [uses GT]", per_img),
                   ("best single thr on test [uses GT]", oracle)]:
        print("%-34s %6.2f %6.2f %6.2f %6.2f %6.2f"
              % ((lbl,) + tuple(100 * r[k] for k in KEYS)))
    print("\ncost of freezing the threshold: %+.2f Dice, %+.2f Jaccard"
          % (100 * (frozen["dice"] - per_img["dice"]),
             100 * (frozen["jaccard"] - per_img["jaccard"])))

    dd = frozen["dice"]
    print("check: Dice %.4f implies Jaccard %.4f, measured %.4f (diff %+.4f)"
          % (dd, dd / (2 - dd), frozen["jaccard"], frozen["jaccard"] - dd / (2 - dd)))


if __name__ == "__main__":
    main()
