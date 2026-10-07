"""
common.py -- data loading, metrics and thresholding shared by the scripts.


"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SIZE = 512


# ---------------------------------------------------------------- dataset

class Retina(Dataset):
    """Image/mask pairs from <root>/Images and <root>/GT, resized to 512x512."""

    def __init__(self, root, augment=False):
        self.img_dir = os.path.join(root, "Images")
        self.gt_dir = os.path.join(root, "GT")
        self.names = sorted(os.listdir(self.img_dir))
        self.gts = sorted(os.listdir(self.gt_dir))
        if len(self.names) != len(self.gts):
            raise SystemExit("%d images but %d masks in %s" % (len(self.names), len(self.gts), root))
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.names[i])).convert("RGB")
        msk = Image.open(os.path.join(self.gt_dir, self.gts[i])).convert("L")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.BILINEAR)
            msk = msk.resize((SIZE, SIZE), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.).permute(2, 0, 1)
        y = torch.from_numpy((np.asarray(msk, np.float32) / 255. > 0.5).astype(np.float32)).unsqueeze(0)
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = torch.flip(x, [2]), torch.flip(y, [2])
            if torch.rand(1) < 0.5:
                x, y = torch.flip(x, [1]), torch.flip(y, [1])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x, y = torch.rot90(x, k, [1, 2]), torch.rot90(y, k, [1, 2])
        return x, y


def load_images(folder):
    """Read a folder of images as a float array in [0,1]; returns (array, names)."""
    names = sorted(os.listdir(folder))
    X = np.zeros((len(names), SIZE, SIZE, 3), np.float32)
    for i, n in enumerate(names):
        X[i] = np.asarray(Image.open(os.path.join(folder, n)).convert("RGB")
                          .resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.
    return X, names


def load_masks(folder):
    names = sorted(os.listdir(folder))
    Y = np.zeros((len(names), SIZE, SIZE), np.float32)
    for i, n in enumerate(names):
        Y[i] = np.asarray(Image.open(os.path.join(folder, n)).convert("L")
                          .resize((SIZE, SIZE), Image.NEAREST), np.float32) / 255. > 0.5
    return Y


# ---------------------------------------------------------------- metrics

def metrics(pred_bin, gt_bin):
    tp = float((pred_bin * gt_bin).sum())
    tn = float(((1 - pred_bin) * (1 - gt_bin)).sum())
    fp = float((pred_bin * (1 - gt_bin)).sum())
    fn = float(((1 - pred_bin) * gt_bin).sum())
    eps = 1e-9
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "jaccard": tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def mean_metrics(per_image):
    return {k: float(np.mean([d[k] for d in per_image])) for k in per_image[0]}


def f1_threshold(prob, gt_bin):
    """Threshold maximising F1 for one image against its own mask."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(gt_bin.ravel(), prob.ravel())
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thresholds[min(i, len(thresholds) - 1)])


def best_global_threshold(probs, gts, lo=0.05, hi=0.995, step=0.005):
    """One threshold for the whole set, chosen on mean Dice."""
    best_t, best_d = 0.5, -1.0
    for t in np.arange(lo, hi, step):
        d = np.mean([metrics((probs[i] >= t).astype(np.float32), gts[i])["dice"]
                     for i in range(len(gts))])
        if d > best_d:
            best_t, best_d = float(t), d
    return best_t, best_d


# ---------------------------------------------------------------- inference

def load_model(weights):
    """Load a checkpoint and return predict_fn(X) -> (N,512,512) probabilities.

    X is a float array of shape (N,512,512,3) in [0,1]. Every checkpoint in
    this repository uses the single PyTorch model in lvs_net.py.
    """
    import torch
    from lvs_net import load_lvs_net
    model, device = load_lvs_net(weights)
    print("model: pytorch (%s)" % device)

    def predict_fn(X):
        out = np.zeros((len(X), SIZE, SIZE), np.float32)
        with torch.no_grad():
            for i in range(len(X)):
                t = torch.from_numpy(np.ascontiguousarray(X[i])).permute(2, 0, 1)[None].to(device)
                out[i] = model(t).cpu().numpy()[0, 0]
        return out

    return predict_fn
