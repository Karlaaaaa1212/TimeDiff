"""
Re-draw / inspect the training curve of a finished (or crashed) run.

    python plot_losses.py                                  # newest run under ./checkpoints
    python plot_losses.py --setting stock_252_21_DDPM_stock_ftM_sl252_ll63_pl21_dt0_TWO
    python plot_losses.py --path ./checkpoints/<setting>/losses.pkl

Prints the per-epoch numbers and writes loss_curve.png next to losses.pkl.
"""

import os
import glob
import pickle
import argparse

import numpy as np

from utils.tools import plot_loss_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='path to losses.pkl')
    parser.add_argument('--setting', type=str, default=None, help='folder name under --checkpoints')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--print_every', type=int, default=1, help='print every k epochs (0 = no table)')
    args = parser.parse_args()

    if args.path is not None:
        path = args.path
    elif args.setting is not None:
        path = os.path.join(args.checkpoints, args.setting, 'losses.pkl')
    else:
        cands = glob.glob(os.path.join(args.checkpoints, '*', 'losses.pkl'))
        if not cands:
            raise SystemExit("No losses.pkl found under {}".format(args.checkpoints))
        path = max(cands, key=os.path.getmtime)
        print("using newest run:", path)

    with open(path, 'rb') as f:
        training_process = pickle.load(f)

    tl = list(training_process.get("train_loss", []))
    vl = list(training_process.get("val_loss", []))
    print("epochs recorded: train={} val={}".format(len(tl), len(vl)))

    if args.print_every > 0 and len(tl) > 0:
        print("\nepoch |   train_loss |     val_mse")
        for e in range(len(tl)):
            if (e + 1) % args.print_every == 0 or e == 0 or e == len(tl) - 1:
                v = vl[e] if e < len(vl) else float('nan')
                print("{:5d} | {:12.7f} | {:11.7f}".format(e + 1, tl[e], v))

    out = os.path.join(os.path.dirname(path), 'loss_curve.png')
    plot_loss_curve(training_process, name=out, title=os.path.basename(os.path.dirname(path)))

    if len(tl) > 1:
        half = len(tl) // 2
        print("\ntrain loss: 1st half mean={:.7f} | 2nd half mean={:.7f}".format(
            float(np.mean(tl[:half])), float(np.mean(tl[half:]))))
    if len(vl) > 1:
        half = len(vl) // 2
        print("val   mse : 1st half mean={:.7f} | 2nd half mean={:.7f} | best epoch={}".format(
            float(np.mean(vl[:half])), float(np.mean(vl[half:])), int(np.argmin(vl)) + 1))


if __name__ == '__main__':
    main()
