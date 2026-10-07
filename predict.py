"""
predict.py -- run a trained LVS-Net checkpoint on a folder of images.

"""

import argparse
import os

import numpy as np
from PIL import Image

from common import (best_global_threshold, f1_threshold, load_images,
                    load_masks, load_model, mean_metrics, metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", default=None, help="ground truth folder; enables metrics")
    ap.add_argument("--out", default=None, help="folder for predicted probability maps")
    args = ap.parse_args()

    X, names = load_images(args.images)
    predict = load_model(args.weights)
    P = predict(X)
    print("predicted %d images" % len(P))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for p, n in zip(P, names):
            Image.fromarray((p * 255).astype(np.uint8)).save(
                os.path.join(args.out, os.path.splitext(n)[0] + "_pred.png"))
        print("probability maps ->", args.out)

    if args.masks:
        Y = load_masks(args.masks)
        r = mean_metrics([metrics((p >= f1_threshold(p, g)).astype(np.float32), g)
                          for p, g in zip(P, Y)])
        best_t, _ = best_global_threshold(P, Y)
        f = mean_metrics([metrics((p >= best_t).astype(np.float32), g)
                          for p, g in zip(P, Y)])
        k = ("dice", "jaccard", "sensitivity", "specificity", "accuracy")
        print("\n%-28s %6s %6s %6s %6s %6s" % ("protocol", "Dice", "J", "Sn", "Sp", "Acc"))
        print("%-28s %6.2f %6.2f %6.2f %6.2f %6.2f"
              % (("per-image F1-optimal",) + tuple(100 * r[x] for x in k)))
        print("%-28s %6.2f %6.2f %6.2f %6.2f %6.2f"
              % (("fixed threshold %.2f" % best_t,) + tuple(100 * f[x] for x in k)))


if __name__ == "__main__":
    main()
