"""
tta.py -- evaluation with test-time augmentation.
"""

import argparse
import os

import numpy as np
import torch

from common import (SIZE, Retina, best_global_threshold, f1_threshold,
                    load_images, load_masks, load_model, mean_metrics, metrics)


def variants(x):
    """8 (transformed image, inverse transform) pairs for one HWC image."""
    out = []
    for flip in (False, True):
        xf = x[:, ::-1] if flip else x
        for k in range(4):
            if flip:
                inv = lambda p, k=k: np.rot90(p, -k, (0, 1))[:, ::-1]
            else:
                inv = lambda p, k=k: np.rot90(p, -k, (0, 1))
            out.append((np.ascontiguousarray(np.rot90(xf, k, (0, 1))), inv))
    return out


def predict_tta(predict, X, n):
    """n = 1 (plain), 4, or 8."""
    P = np.zeros((len(X), SIZE, SIZE), np.float32)
    for i, x in enumerate(X):
        vs = variants(x)[:n] if n > 1 else [(x, lambda p: p)]
        preds = predict(np.stack([v for v, _ in vs]))
        acc = np.zeros((SIZE, SIZE), np.float32)
        for p, (_, inv) in zip(preds, vs):
            acc += inv(p)
        P[i] = acc / len(vs)
    return P


def report(name, P, G, Pva, Gva):
    t_val, _ = best_global_threshold(Pva, Gva)
    f1 = mean_metrics([metrics((p >= f1_threshold(p, g)).astype(np.float32), g)
                       for p, g in zip(P, G)])
    val = mean_metrics([metrics((p >= t_val).astype(np.float32), g)
                        for p, g in zip(P, G)])
    print("%-12s F1-thr: Dice %.2f  J %.2f  Sn %.2f  Sp %.2f   |  val-thr %.2f: Dice %.2f  (gap %.2f)"
          % (name, 100 * f1["dice"], 100 * f1["jaccard"], 100 * f1["sensitivity"],
             100 * f1["specificity"], t_val, 100 * val["dice"],
             100 * (f1["dice"] - val["dice"])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="folder holding Train/ and Test/")
    ap.add_argument("--save", default=None, help="save the 8-way TTA maps to this .npy")
    args = ap.parse_args()

    predict = load_model(args.weights)

    Xte, _ = load_images(os.path.join(args.data, "Test", "Images"))
    Yte = load_masks(os.path.join(args.data, "Test", "GT"))

    # the same validation split train.py / finetune.py hold out
    full = Retina(os.path.join(args.data, "Train"))
    n_val = max(1, len(full) // 10)
    _, va = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    Xva = np.stack([full[i][0].permute(1, 2, 0).numpy() for i in va.indices])
    Gva = np.stack([full[i][1][0].numpy() for i in va.indices])
    print("weights: %s | test %d | val %d" % (args.weights, len(Xte), len(Xva)))

    for n in (1, 4, 8):
        Pva = predict_tta(predict, Xva, n)
        P = predict_tta(predict, Xte, n)
        report("plain" if n == 1 else "%d-way TTA" % n, P, Yte, Pva, Gva)
        if n == 8 and args.save:
            np.save(args.save, P)
            print("8-way TTA maps ->", args.save)


if __name__ == "__main__":
    main()
