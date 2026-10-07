"""
evaluate.py -- full per-image evaluation of a checkpoint on a test set.

"""

import argparse
import os

import numpy as np

from common import (best_global_threshold, f1_threshold, load_images,
                    load_masks, load_model, mean_metrics, metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="folder holding Images/ and GT/")
    ap.add_argument("--compare", nargs=4, type=float, default=None,
                    metavar=("DICE", "JACC", "SN", "SP"),
                    help="published values to compare against")
    args = ap.parse_args()

    X, names = load_images(os.path.join(args.data, "Images"))
    Y = load_masks(os.path.join(args.data, "GT"))
    print("loaded %d test images from %s" % (len(X), args.data))

    predict = load_model(args.weights)
    P = predict(X)

    per, thr = [], []
    for i in range(len(Y)):
        t = f1_threshold(P[i], Y[i])
        thr.append(t)
        per.append(metrics((P[i] >= t).astype(np.float32), Y[i]))
    f1_result = mean_metrics(per)

    best_t, _ = best_global_threshold(P, Y)
    fixed = mean_metrics([metrics((P[i] >= best_t).astype(np.float32), Y[i])
                          for i in range(len(Y))])

    print("\nper-image results (F1-optimal threshold)")
    print("  %-22s %7s %7s %7s %7s %6s" % ("image", "Dice", "Jacc", "Sn", "Sp", "thr"))
    for i, n in enumerate(names):
        m = per[i]
        print("  %-22s %7.2f %7.2f %7.2f %7.2f %6.2f"
              % (n[:22], 100 * m["dice"], 100 * m["jaccard"],
                 100 * m["sensitivity"], 100 * m["specificity"], thr[i]))

    print("\n%-26s %7s %7s %7s %7s" % ("protocol", "Dice", "Jacc", "Sn", "Sp"))
    print("-" * 58)
    for label, r in [("per-image F1-optimal", f1_result),
                     ("single fixed thr=%.2f" % best_t, fixed)]:
        print("%-26s %7.2f %7.2f %7.2f %7.2f"
              % (label, 100 * r["dice"], 100 * r["jaccard"],
                 100 * r["sensitivity"], 100 * r["specificity"]))

    if args.compare:
        d, j, s, p = args.compare
        print("%-26s %7.2f %7.2f %7.2f %7.2f" % ("published", d, j, s, p))
        print("%-26s %+7.2f %+7.2f %+7.2f %+7.2f"
              % ("difference",
                 100 * f1_result["dice"] - d, 100 * f1_result["jaccard"] - j,
                 100 * f1_result["sensitivity"] - s, 100 * f1_result["specificity"] - p))

    # Dice and Jaccard must satisfy IoU = Dice / (2 - Dice) per image; after
    # averaging, Jensen's inequality allows the mean Jaccard to sit slightly
    # above the value implied by the mean Dice, never below.
    dd = f1_result["dice"]
    print("\ncheck: Dice %.4f implies Jaccard %.4f, measured %.4f (diff %+.4f)"
          % (dd, dd / (2 - dd), f1_result["jaccard"], f1_result["jaccard"] - dd / (2 - dd)))


if __name__ == "__main__":
    main()
