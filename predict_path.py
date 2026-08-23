"""
One continuous forecast path over the whole test period.

Instead of the 1216 overlapping 21-day forecasts that Exp_Main.test() scores, this walks
the test period in NON-OVERLAPPING steps -- forecast 21 days, jump 21 days, forecast
again -- and joins the blocks end to end.  The result is a single daily series per
ticker covering 2020-2024, and a single compounded price path built from it:

    P_hat[t] = P_0 * exp( cumsum(r_hat) )        P_0 = the real close before day one

Nothing is re-anchored along the way: block 2 starts where block 1's generated path
ended, so any drift bias compounds -- which is the point of looking at it this way.

The test period is rarely a whole number of blocks: with 63/21 the stride-21 walk stops
at 2024-12-03 and leaves 19 days over.  So one extra block is anchored at the very end
of the data, and only the days it adds beyond the previous block are kept -- the path
then runs to the last test day with no gap and no duplicated dates.

With --sample_times S every diffusion sample is chained into its own continuous path and
each one is written out separately, numbered path_0001, path_0002, ....  Path 1 is the one the
figure draws and the one the top-level pred_*.csv hold; the rest are there for whatever
you want to compute over the ensemble.  A 10-90% band across all S paths is shaded in.

    python predict_path.py <model flags> [--tag v --parameterization v] [--sample_times 500]
    python predict_path.py <model flags> --path_knob -3 --knob_guidance 5 --sample_times 500

Each (knob, guidance) pair writes to its own directory -- path_forecast/,
path_forecast_knob-3/, path_forecast_knob-3_w5/ -- so the runs never overwrite.

Writes into <setting>/path_forecast/, all in long form  date, tic, close, high, low:

    paths_simple_returns/path_0001.csv ... one file per generated path
    paths_prices/path_0001.csv         ... one file per generated path
    true_simple_returns.csv / true_prices.csv   only in the no-knob directory:
                                                the truth is the same for every scenario
        close = C_t/C_{t-1} - 1   the day-over-day simple return
        high  = H_t/C_t - 1       intraday upside, relative to the SAME day's close
        low   = L_t/C_t - 1       intraday downside, same convention  (<= 0)

    pred_prices.csv / true_prices.csv
        close = C_0 * exp(cumsum(r_close))    compounded from the last real close
        high  = C_t * exp(r_high)             rebuilt off that day's close
        low   = C_t * exp(r_low)

    stitched_path.png   one panel per ticker, real close vs the continuous generated one

The source file stores log returns, so r_high/r_low are ratios to the same day's close,
not day-over-day moves -- they cannot be compounded, only applied to C_t.
"""

import os
import math
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from config import get_args
from exp.exp_main import Exp_Main
from data_provider.data_factory import data_provider
from plot_price import build_setting, predict_windows

C_REAL, C_PRED = '#1f77b4', '#d62728'


def path_args(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--path_knob', type=float, default=None,
                   help='the extreme-scenario dial, in sigma of the cumulative move over '
                        'the horizon. Omit for the plain no-knob run. Not --knob: that one '
                        'belongs to config.py and is read by the model itself.')
    p.add_argument('--price_file', type=str, default='stock_adjclose.csv')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--flag', type=str, default='test', choices=['test', 'val', 'train'])
    a, _ = p.parse_known_args(argv)
    return a


def main():
    popt = path_args()
    args = get_args(known_only=True)
    H, SL = args.pred_len, args.seq_len

    setting = build_setting(args)
    ckpt_dir = os.path.join(args.checkpoints, setting)

    # each (knob, guidance) combination gets its own directory so the runs never collide
    name = 'path_forecast'
    if popt.path_knob is not None:
        name += '_knob%g' % popt.path_knob
        g = float(getattr(args, 'knob_guidance', 1.0))
        if g != 1.0:
            name += '_w%g' % g
    out_dir = popt.out_dir or os.path.join(ckpt_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    exp = Exp_Main(args)
    exp.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
    dataset, _ = data_provider(args, flag=popt.flag)
    dates = pd.to_datetime(dataset.dates)

    # walk the test period in strides of pred_len: forecast 21, jump 21, forecast again
    N = len(dataset.data_x)
    window_ids = [i for i in range(0, len(dataset), H) if i + SL + H <= N]
    take = [H] * len(window_ids)                     # days used from each block

    # the leftover at the end is shorter than a block, so anchor one final window on the
    # last day and keep only the days it adds
    covered = window_ids[-1] + SL + H
    if covered < N:
        window_ids.append(N - SL - H)
        take.append(N - covered)

    print('setting : %s' % setting)
    print('output  : %s' % out_dir)
    print('knob    : %s | guidance %g'
          % ('none' if popt.path_knob is None else '%g' % popt.path_knob,
             float(getattr(args, 'knob_guidance', 1.0))))
    print('stride  : %d days | %d blocks (last one adds %d days) | samples %d'
          % (H, len(window_ids), take[-1], args.sample_times))

    preds = predict_windows(exp, dataset, window_ids, args, knob=popt.path_knob)
    W, S, L, V = preds.shape

    ret = pd.read_csv(os.path.join(args.root_path, args.data_path),
                      parse_dates=['date']).set_index('date')
    px = pd.read_csv(os.path.join(args.root_path, popt.price_file),
                     parse_dates=['date']).set_index('date')
    close_cols = [c for c in ret.columns if c.endswith('_r_close')]
    tics = [c[:-len('_r_close')] for c in close_cols]
    dcols = list(dataset.cols)
    cidx = [dcols.index(t + '_r_close') for t in tics]
    hidx = [dcols.index(t + '_r_high') for t in tics]
    lidx = [dcols.index(t + '_r_low') for t in tics]

    # the blocks are contiguous in time, so concatenating them IS the stitched path.
    # `take` trims the overlap the final block has with the one before it.
    fut = [dates[i + SL: i + SL + H][H - k:] for i, k in zip(window_ids, take)]
    path_dates = pd.DatetimeIndex(np.concatenate(fut))
    r_all = np.concatenate([preds[w][:, H - k:, :] for w, k in enumerate(take)], axis=1)
    assert len(path_dates) == r_all.shape[1] == len(set(path_dates))
    r_pred, h_pred, l_pred = r_all[:, :, cidx], r_all[:, :, hidx], r_all[:, :, lidx]
    r_true = ret.loc[path_dates, [t + '_r_close' for t in tics]].values
    h_true = ret.loc[path_dates, [t + '_r_high' for t in tics]].values
    l_true = ret.loc[path_dates, [t + '_r_low' for t in tics]].values
    print('path    : %s ~ %s  (%d days)'
          % (path_dates[0].date(), path_dates[-1].date(), len(path_dates)))

    # compound each sample into its own price path, seeded once on the real close
    anchor_date = dates[window_ids[0] + SL - 1]
    p0 = px.loc[anchor_date, tics].values                               # (V,)
    p_pred = p0 * np.exp(np.cumsum(r_pred, axis=1))                     # (S, W*L, V)
    p_true = px.loc[path_dates, tics].values

    e = lambda a: np.exp(a) - 1.0                       # log return -> simple return
    n = len(path_dates)
    _date = np.repeat(path_dates.strftime('%Y-%m-%d').values, len(tics))
    _tic = np.tile(tics, n)

    def frame(close, high, low):
        """long form: one row per (date, tic)"""
        return pd.DataFrame({'date': _date, 'tic': _tic, 'close': close.reshape(-1),
                             'high': high.reshape(-1), 'low': low.reshape(-1)})

    def dump(path, close, high, low, quiet=False):
        frame(close, high, low).to_csv(path, index=False, float_format='%.8g')
        if not quiet:
            print('saved %-34s %d rows' % (os.path.relpath(path, out_dir), n * len(tics)))

    # ---- every generated path, one numbered file each ----
    dir_r = os.path.join(out_dir, 'paths_simple_returns')
    dir_p = os.path.join(out_dir, 'paths_prices')
    os.makedirs(dir_r, exist_ok=True); os.makedirs(dir_p, exist_ok=True)
    n_clipped = 0
    for i in range(S):
        name = 'path_%04d.csv' % (i + 1)
        dump(os.path.join(dir_r, name), e(r_pred[i]), e(h_pred[i]), e(l_pred[i]), quiet=True)

        close_i = p_pred[i]
        high_i = p_pred[i] * np.exp(h_pred[i])
        low_i = p_pred[i] * np.exp(l_pred[i])
        # h_pred/l_pred are sampled independently of r_pred, so nothing stops a draw
        # from landing high < close or low > close. Clip to the nearest side of close --
        # the minimal correction that restores low <= close <= high without touching
        # close, the value everything downstream (portfolio value, returns) is keyed on.
        bad = (low_i > close_i) | (close_i > high_i) | (low_i > high_i)
        n_clipped += int(bad.sum())
        high_i = np.maximum(high_i, close_i)
        low_i = np.minimum(low_i, close_i)

        dump(os.path.join(dir_p, name), close_i, high_i, low_i, quiet=True)
        if (i + 1) % 100 == 0 or i + 1 == S:
            print('  wrote %d/%d paths' % (i + 1, S), flush=True)
    print('saved %-34s %d files each' % ('paths_simple_returns/ , paths_prices/', S))
    if n_clipped:
        pct = 100.0 * n_clipped / (S * n * len(tics))
        print('  paths_prices/: clipped %d/%d rows (%.1f%%) that violated low<=close<=high '
              '(high/low pulled to close; paths_simple_returns/ is left unclipped, raw model output)'
              % (n_clipped, S * n * len(tics), pct))

    # path_0001 IS the representative path, so no pred_*.csv copy of it is written.
    # The ground truth does not depend on the model or the knob, so it is written once,
    # in the no-knob directory, instead of once per scenario.
    if popt.path_knob is None:
        dump(os.path.join(out_dir, 'true_simple_returns.csv'), e(r_true), e(h_true), e(l_true))
        dump(os.path.join(out_dir, 'true_prices.csv'), p_true,
             p_true * np.exp(h_true), p_true * np.exp(l_true))

    # ---- the figure: one continuous line per ticker ----
    ncol = 5
    nrow = int(math.ceil(len(tics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.3 * nrow), squeeze=False)
    for v in range(nrow * ncol):
        ax = axes[v // ncol][v % ncol]
        if v >= len(tics):
            ax.axis('off'); continue
        if S > 1:
            ax.fill_between(path_dates, np.percentile(p_pred[:, :, v], 10, axis=0),
                            np.percentile(p_pred[:, :, v], 90, axis=0),
                            color=C_PRED, alpha=0.20, lw=0,
                            label='10-90%% of %d paths' % S if v == 0 else None)
        ax.plot(path_dates, p_true[:, v], '-', c=C_REAL, lw=1.4,
                label='real' if v == 0 else None)
        ax.plot(path_dates, p_pred[0, :, v], '-', c=C_PRED, lw=1.4,
                label='forecast (path 1)' if v == 0 else None)
        end = 100 * (p_pred[0, -1, v] / p_true[-1, v] - 1)
        ax.set_title('%s   ends %+.0f%% vs real' % (tics[v], end), fontsize=11)
        ax.grid(alpha=0.3); ax.tick_params(axis='x', rotation=30, labelsize=8)
        if v % ncol == 0:
            ax.set_ylabel('adjusted close (USD)')
        if v == 0:
            ax.legend(fontsize=8)
    scen = ('no knob' if popt.path_knob is None else
            'knob %g, guidance %g' % (popt.path_knob, float(getattr(args, 'knob_guidance', 1.0))))
    fig.suptitle('One continuous path, %d-day blocks joined end to end   %s ~ %s\n%s   [%s]'
                 % (H, path_dates[0].date(), path_dates[-1].date(), setting, scen),
                 y=1.01, fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(out_dir, 'stitched_path.png')
    fig.savefig(out, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('saved', out)

    # ---- the same volatility / distribution checks, on the stitched series ----
    from plot_test_analysis import make_figures
    full = [w for w, k in enumerate(take) if k == H]     # the analysis wants equal blocks
    blocks = [dates[window_ids[w] + SL: window_ids[w] + SL + H] for w in full]
    make_figures(out_dir, preds[full][:, 0],          # path 1, NOT the sample mean
                 np.stack([ret.loc[b, ret.columns].values for b in blocks]),
                 [str(c) for c in dataset.cols],
                 dates=np.stack([[d.strftime('%Y-%m-%d') for d in b] for b in blocks]),
                 tag='  [stride %d, %s]' % (H, scen), prefix='path_')


if __name__ == '__main__':
    main()
