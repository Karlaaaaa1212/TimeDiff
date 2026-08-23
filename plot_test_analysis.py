"""
The two figures every test run has to produce.

  <setting>/volatility_check.png              all ten tickers on one axis
  <setting>/volatility_check_by_ticker.png    one panel per ticker, through time
  <setting>/distribution_check.png            the pooled distribution
  <setting>/distribution_check_by_ticker.png  one panel per ticker

Exp_Main.test() calls make_figures() automatically at the end of every test, so these
appear next to the metrics without asking.  The same hook writes the predictions
themselves next to them:

  <setting>/test_predictions.csv    one row per (window, forecast day), one column per
  <setting>/test_ground_truth.csv   variable, with the calendar date of each row
  <setting>/test_arrays.npz         the same thing as arrays: preds, trues, cols, dates

All of it is in real log-return units (inverse_transform applied), not sigma units, so
the figures can be redrawn offline at any time -- no GPU, no checkpoint:

    python plot_test_analysis.py --setting stock_63_21_DDPM_stock_ftM_sl63_ll21_pl21_dt0_TWO

Everything is scaled by each ticker's REALIZED test volatility, so the real series has
sd = 1.0 by construction and the model's sd reads directly as "fraction of the market's
volatility".
"""

import os
import math
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

TAB10 = plt.get_cmap('tab10').colors
C_REAL, C_PRED = '#1f77b4', '#d62728'
ANN = math.sqrt(252) * 100          # daily -> annualized %


def _close_channels(cols, n_vars):
    """-> (labels, indices). Prefers the *_r_close columns when the file has them."""
    cols = [str(c) for c in cols]
    close = [(c[:-len('_r_close')], i) for i, c in enumerate(cols) if c.endswith('_r_close')]
    if close:
        return [c[0] for c in close], [c[1] for c in close]
    if cols:
        return cols, list(range(len(cols)))
    return ['var%d' % i for i in range(n_vars)], list(range(n_vars))


def volatility_figure(labels, real, pred, out_png, title=''):
    """
    real / pred : (n_windows, pred_len, n_tickers) log returns in real units
    """
    W, L, V = real.shape
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # (a) level, all tickers on one axis
    ax = axes[0]
    rv = real.reshape(-1, V).std(axis=0) * ANN
    pv = pred.std(axis=1).mean(axis=0) * ANN          # vol inside a window, then averaged
    x = np.arange(V)
    ax.bar(x - 0.2, rv, 0.4, color=C_REAL, label='realized')
    ax.bar(x + 0.2, pv, 0.4, color=C_PRED, label='predicted')
    for i in range(V):
        ax.text(i + 0.2, pv[i] + rv.max() * 0.03, '%.0f%%' % (100 * pv[i] / rv[i]),
                ha='center', fontsize=8, color=C_PRED, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60)
    ax.set_ylabel('annualized volatility (%)')
    ax.set_title('(a) level  -  model reaches %.0f%% of the market' % (100 * np.mean(pv / rv)))
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # (b) does it track the volatility of each individual window?
    ax = axes[1]
    rw = real.std(axis=1) * ANN                        # (W, V)
    pw = pred.std(axis=1) * ANN
    for v in range(V):
        ax.scatter(rw[:, v], pw[:, v], s=9, color=TAB10[v % 10], alpha=0.55, label=labels[v])
    hi = max(rw.max(), pw.max()) * 1.05
    ax.plot([0, hi], [0, hi], 'k--', lw=1.2)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel('realized vol of the window (%, ann.)')
    ax.set_ylabel('predicted vol of the window (%, ann.)')
    ax.set_title('(b) tracking  -  corr = %.2f' % np.corrcoef(rw.ravel(), pw.ravel())[0, 1])
    ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.3)

    # (c) through time, averaged across tickers
    ax = axes[2]
    t = np.arange(W)
    ax.plot(t, rw.mean(axis=1), '-', c=C_REAL, lw=1.6, label='realized')
    ax.plot(t, pw.mean(axis=1), '-', c=C_PRED, lw=1.6, label='predicted')
    ax.fill_between(t, np.percentile(rw, 25, axis=1), np.percentile(rw, 75, axis=1),
                    color=C_REAL, alpha=0.15, lw=0)
    ax.fill_between(t, np.percentile(pw, 25, axis=1), np.percentile(pw, 75, axis=1),
                    color=C_PRED, alpha=0.15, lw=0)
    ax.set_xlabel('test window (chronological)')
    ax.set_ylabel('%d-day volatility (%%, ann.)' % L)
    ax.set_title('(c) through time  -  band = 25-75% across tickers')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Volatility: predicted vs realized%s   (%d windows x %d days x %d tickers)'
                 % (title, W, L, V), y=1.02)
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('saved', out_png)


def distribution_figure(labels, real, pred, out_png, title=''):
    W, L, V = real.shape
    sd = real.reshape(-1, V).std(axis=0)               # each ticker's own realized sd
    zr, zp = real / sd, pred / sd                      # real sd == 1 by construction

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    # (a) the body of the distribution
    ax = axes[0, 0]
    bins = np.linspace(-6, 6, 121)
    ax.hist(zr.ravel(), bins=bins, density=True, color=C_REAL, alpha=0.55, label='real')
    ax.hist(zp.ravel(), bins=bins, density=True, color=C_PRED, alpha=0.55, label='predicted')
    ax.set_xlim(-6, 6); ax.set_xlabel('daily return / realized sigma')
    ax.set_title('(a) the distribution   real sd 1.00  vs  model sd %.2f' % zp.std())
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) the tails
    ax = axes[0, 1]
    ax.hist(zr.ravel(), bins=bins, density=True, histtype='step', lw=1.7, color=C_REAL, label='real')
    ax.hist(zp.ravel(), bins=bins, density=True, histtype='step', lw=1.7, color=C_PRED, label='predicted')
    g = np.linspace(-6, 6, 300)
    ax.plot(g, np.exp(-g ** 2 / 2) / math.sqrt(2 * math.pi), 'k--', lw=1.1, label='N(0,1)')
    ax.set_yscale('log'); ax.set_ylim(1e-5, 1); ax.set_xlim(-6, 6)
    ax.set_xlabel('daily return / realized sigma')
    ax.set_title('(b) the tails (log scale)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) quantile against quantile
    ax = axes[1, 0]
    q = np.linspace(0.001, 0.999, 400)
    ax.plot(np.quantile(zr.ravel(), q), np.quantile(zp.ravel(), q), '-', c=C_PRED, lw=2)
    lo, hi = np.quantile(zr.ravel(), [0.001, 0.999])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2)
    ax.set_xlabel('real quantile'); ax.set_ylabel('predicted quantile')
    ax.set_title('(c) Q-Q  -  on the line = same distribution')
    ax.grid(alpha=0.3)

    # (d) shape, per ticker
    ax = axes[1, 1]
    kr = [float(((zr[:, :, v] - zr[:, :, v].mean()) ** 4).mean() /
                (zr[:, :, v].var() ** 2) - 3) for v in range(V)]
    kp = [float(((zp[:, :, v] - zp[:, :, v].mean()) ** 4).mean() /
                (zp[:, :, v].var() ** 2) - 3) for v in range(V)]
    x = np.arange(V)
    ax.bar(x - 0.2, kr, 0.4, color=C_REAL, label='real')
    ax.bar(x + 0.2, kp, 0.4, color=C_PRED, label='predicted')
    ax.axhline(0, color='k', lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60)
    ax.set_ylabel('excess kurtosis')
    ax.set_title('(d) fat tails per ticker  (0 = gaussian)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('Distribution: predicted vs real%s   (%d windows x %d days x %d tickers)'
                 % (title, W, L, V), y=1.00)
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('saved', out_png)


def _grid(V):
    """(rows, cols) for V small multiples -- 10 tickers land on 2 x 5"""
    ncol = 5 if V > 6 else max(V, 1)
    return int(math.ceil(V / ncol)), ncol


def volatility_by_ticker(labels, real, pred, out_png, dates=None, title=''):
    """One panel per ticker: the volatility of every forecast window, through time."""
    W, L, V = real.shape
    rw, pw = real.std(axis=1) * ANN, pred.std(axis=1) * ANN     # (W, V)
    x = pd.to_datetime(dates) if dates is not None else np.arange(W)

    nrow, ncol = _grid(V)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow),
                             sharex=True, squeeze=False)
    for v in range(nrow * ncol):
        ax = axes[v // ncol][v % ncol]
        if v >= V:
            ax.axis('off'); continue
        ax.plot(x, rw[:, v], '-', c=C_REAL, lw=1.1, label='realized')
        ax.plot(x, pw[:, v], '-', c=C_PRED, lw=1.3, label='predicted')
        ax.set_title('%s   %.0f%% of realized' % (labels[v], 100 * pw[:, v].mean() / rw[:, v].mean()),
                     fontsize=11)
        ax.grid(alpha=0.3)
        if v % ncol == 0:
            ax.set_ylabel('%d-day vol (%%, ann.)' % L)
        if v == 0:
            ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=30, labelsize=8)

    fig.suptitle('Volatility per ticker: predicted vs realized%s' % title, y=1.005, fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('saved', out_png)


def distribution_by_ticker(labels, real, pred, out_png, title=''):
    """One panel per ticker: its own return distribution, log scale so the tails show."""
    W, L, V = real.shape
    sd = real.reshape(-1, V).std(axis=0)
    zr, zp = real / sd, pred / sd
    bins = np.linspace(-6, 6, 97)
    g = np.linspace(-6, 6, 300)
    gauss = np.exp(-g ** 2 / 2) / math.sqrt(2 * math.pi)

    nrow, ncol = _grid(V)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for v in range(nrow * ncol):
        ax = axes[v // ncol][v % ncol]
        if v >= V:
            ax.axis('off'); continue
        a, b = zr[:, :, v].ravel(), zp[:, :, v].ravel()
        ax.hist(a, bins=bins, density=True, histtype='step', lw=1.6, color=C_REAL, label='real')
        ax.hist(b, bins=bins, density=True, histtype='step', lw=1.6, color=C_PRED, label='predicted')
        ax.plot(g, gauss, 'k--', lw=1.0)
        ax.set_yscale('log'); ax.set_ylim(1e-4, 3); ax.set_xlim(-6, 6)
        k = float(((b - b.mean()) ** 4).mean() / (b.var() ** 2) - 3)
        kr = float(((a - a.mean()) ** 4).mean() / (a.var() ** 2) - 3)
        ax.set_title('%s   sd %.2f   kurt %.1f / %.1f' % (labels[v], b.std(), kr, k), fontsize=10.5)
        ax.grid(alpha=0.3)
        if v % ncol == 0:
            ax.set_ylabel('density (log)')
        if v // ncol == nrow - 1:
            ax.set_xlabel('return / realized sigma')
        if v == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Distribution per ticker (dashed = N(0,1); titles show real/predicted kurtosis)%s'
                 % title, y=1.005, fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('saved', out_png)


def make_figures(folder, preds, trues, cols, dates=None, tag='', prefix=''):
    """preds/trues: (n_windows, pred_len, n_vars) log returns in real units."""
    labels, idx = _close_channels(cols, preds.shape[-1])
    real, pred = trues[:, :, idx], preds[:, :, idx]
    d0 = None
    if dates is not None and len(np.asarray(dates)):
        d0 = np.asarray(dates)[:, 0]          # the first forecast day of each window
    at = lambda n: os.path.join(folder, prefix + n)
    volatility_figure(labels, real, pred, at('volatility_check.png'), tag)
    volatility_by_ticker(labels, real, pred, at('volatility_check_by_ticker.png'), d0, tag)
    distribution_figure(labels, real, pred, at('distribution_check.png'), tag)
    distribution_by_ticker(labels, real, pred, at('distribution_check_by_ticker.png'), tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--setting', type=str, required=True)
    p.add_argument('--checkpoints', type=str, default='./checkpoints/')
    a = p.parse_args()
    folder = os.path.join(a.checkpoints, a.setting)
    z = np.load(os.path.join(folder, 'test_arrays.npz'), allow_pickle=True)
    make_figures(folder, z['preds'], z['trues'], list(z['cols']),
                 dates=z['dates'] if 'dates' in z.files else None)


if __name__ == '__main__':
    main()
