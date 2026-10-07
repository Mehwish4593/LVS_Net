"""
finetune.py -- stage 2: continue training with a cosine-decayed learning rate.

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
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True, help="checkpoint to continue from")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4, help="restart LR, decayed to ~0")
    ap.add_argument("--out", default="runs/finetuned")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = Retina(os.path.join(args.data, "Train"), augment=True)
    n_val = max(1, len(full) // 10)
    tr, va = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6,
                    drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch, num_workers=2)

    model = LVSNet().to(device)
    model.load_state_dict(torch.load(args.init, map_location=device))
    print("resumed from", args.init)

    P, G = run_loader(model, vl, device)
    _, start = best_global_threshold(P, G, step=0.05)
    print("starting val Dice: %.4f" % start)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(args.out, exist_ok=True)

    best, best_ep = start, 0
    torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

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
        sched.step()

        P, G = run_loader(model, vl, device)
        _, vd = best_global_threshold(P, G, step=0.05)

        flag = ""
        if vd > best:
            best, best_ep = vd, ep
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            flag = "  <- best"
        if ep % 10 == 0 or flag or ep == 1:
            print("ep %3d/%d  loss %.4f  val %.4f  lr %.2e  (%.0fs)%s"
                  % (ep, args.epochs, tot / len(tl), vd,
                     sched.get_last_lr()[0], time.time() - t0, flag), flush=True)

    print("\nbest val %.4f at epoch %d (started from %.4f)  ->  %s/best.pt"
          % (best, best_ep, start, args.out))


if __name__ == "__main__":
    main()
