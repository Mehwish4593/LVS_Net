"""
train.py -- stage 1: train LVS-Net from scratch.


"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import Retina, best_global_threshold
from lvs_net import LVSNet, dice_loss


@torch.no_grad()
def run_loader(model, loader, device):
    model.eval()
    P, G = [], []
    for x, y in loader:
        P.append(model(x.to(device)).cpu().numpy()[:, 0])
        G.append(y.numpy()[:, 0])
    return np.concatenate(P), np.concatenate(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding Train/ and Test/")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="runs/train")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    full = Retina(os.path.join(args.data, "Train"))
    n_val = max(1, int(len(full) * args.val_frac))
    tr, va = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    print("train %d | val %d" % (len(tr), len(va)))

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch, num_workers=2)

    model = LVSNet().to(device)
    print("parameters: %s" % format(sum(p.numel() for p in model.parameters()), ","))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)

    best, best_ep = -1.0, -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = dice_loss(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()

        P, G = run_loader(model, vl, device)
        _, vdice = best_global_threshold(P, G, step=0.05)

        flag = ""
        if vdice > best:
            best, best_ep = vdice, ep
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            flag = "  <- best"
        print("ep %3d/%d  loss %.4f  val %.4f  (%.0fs)%s"
              % (ep, args.epochs, tot / len(tl), vdice, time.time() - t0, flag),
              flush=True)

    print("\nbest val Dice %.4f at epoch %d  ->  %s/best.pt" % (best, best_ep, args.out))


if __name__ == "__main__":
    main()
