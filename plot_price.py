"""
Two figures per ticker, both showing History / GroundTruth / Prediction:

  <TIC>_logret.png   close log return   r = log(C_t / C_{t-1})   (real units, not standardized)
  <TIC>_price.png    price rebuilt from that return
                            P_hat[t+k] = P[t] * exp(sum_{j<=k} r_hat[t+j])
                            P[t] = the real adjusted close on the last day of the input window

That re-anchoring is what --price_chain 0 does.  By default (--price_chain 1) the
generated line is instead ONE continuous compounded path: only the first block is
seeded with a real close, every later block starts exactly where the previous
generated block ended, and the returns keep compounding on top of each other, so
the line never snaps back to the real price and never breaks between blocks.
Chaining compounds whatever per-block drift bias the model has, so over a long
range the line can run far away from reality -- that is the point of drawing it.
The summary then carries a "chain MAPE%" column for the continuous path;
"price MAPE%" is always the re-anchored, per-block number.

Only the <TIC>_r_close channel is drawn; r_high / r_low are intraday ranges
relative to the same day's close, so they cannot be compounded into a price.

A geometric Brownian motion baseline (green) is drawn next to the model.  Its
drift mu and volatility sigma are estimated from the same input window the model
sees, then pred_len i.i.d. gaussian returns are simulated per path:
      r_hat[t+j] ~ N(mu, sigma^2)      P_hat[t+k] = P[t] * exp(sum_{j<=k} r_hat)
Its MAPE / direction hit rate are reported next to the model's, so the table
says whether the diffusion model beats a random walk.  --gbm_drift zero pins
mu = 0 (pure driftless random walk); --gbm 0 turns the baseline off.

--knob_values adds the extreme-scenario line: the SAME weights, the same input
window, but the knob dialed to k sigma of cumulative move over the horizon.  The
figure then carries the four lines you want to compare -- History (black),
GroundTruth (blue), Prediction with the original two conditions (red), and
Prediction with the knob (orange).  Nothing on any of those lines reads the
future; the knob is a number you choose.

Give it exactly the same flags as the training run, plus the plotting options:

    # one forecast window
    python plot_price.py --dataset_name stock --seq_len 63 --label_len 21 --pred_len 21 \
        --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 \
        --ablation_study_F_type Linear --cond_ddpm_num_layers 30 --ddpm_layers_II 10 \
        --plot_mode window --plot_window 0

    # the whole test period, stitched from non-overlapping windows
    python plot_price.py ... --plot_mode year

    # with an uncertainty band (needs several diffusion samples per window)
    python plot_price.py ... --plot_mode year --sample_times 20

    # knob comparison, four lines, GBM off
    python plot_price.py ... --plot_mode year --gbm 0 --knob_values=-3

    # does the knob do anything? sweep it and read the monotonicity table
    python plot_price.py ... --plot_mode year --gbm 0 --knob_values=-3,-1,0,1,3

    # only the COVID crash
    python plot_price.py ... --test_start 2020-02-19 --test_end 2020-03-23 \
        --plot_mode window --plot_window 0 --gbm 0 --knob_values=-3
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import get_args
from exp.exp_main import Exp_Main
from data_provider.data_factory import data_provider


def plot_args(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--plot_mode', type=str, default='window', choices=['window', 'year'],
                   help="window: a single forecast window | year: non-overlapping windows over the whole test period")
    p.add_argument('--plot_window', type=int, default=0, help='window index for --plot_mode window')
    p.add_argument('--history_days', type=int, default=-1,
                   help='days of real history to draw; -1 = the whole input window (seq_len)')
    p.add_argument('--plot_band', type=int, default=0,
                   help='1 = shade the 10-90%% range of the diffusion samples (needs --sample_times > 1)')
    p.add_argument('--gbm', type=int, default=1, help='1 = draw the geometric Brownian motion baseline')
    p.add_argument('--gbm_paths', type=int, default=2000, help='simulated paths per window')
    p.add_argument('--gbm_drift', type=str, default='hist', choices=['hist', 'zero'],
                   help="hist: mu = mean return of the input window | zero: driftless random walk")
    p.add_argument('--gbm_seed', type=int, default=0)
    p.add_argument('--gbm_show_paths', type=int, default=0,
                   help='also draw this many individual simulated paths (thin green)')
    p.add_argument('--knob_values', type=str, default='',
                   help='comma separated knob settings to overlay, e.g. "-3" or "-3,-2,-1". '
                        'Each one adds a 4th line: the same model asked for a k-sigma move '
                        'over the horizon. Empty = no knob line.')
    p.add_argument('--price_chain', type=int, default=1,
                   help='1 (default) = one continuous generated path: only the FIRST block is '
                        'seeded with a real close and each later block starts where the previous '
                        'generated block ended, so the line keeps compounding and never snaps '
                        'back to the real price or breaks between blocks. 0 = re-anchor every '
                        'block on the real close of its own last input day (the line then jumps '
                        'at every block boundary). Chaining compounds any per-block drift bias, '
                        'so the line can run far away from reality over a long range.')
    p.add_argument('--price_file', type=str, default='stock_adjclose.csv')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--flag', type=str, default='test', choices=['test', 'val', 'train'])
    a, _ = p.parse_known_args(argv)
    return a


def build_setting(args):
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dt{}_{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.ii, args.stage_mode)
    if args.tag != '':
        setting += '_' + str(args.tag)
    return setting


def window_item(dataset, i):
    """
    The window starting at row `i`, with the decoder part padded when it runs
    past the end of the range.

    The rows are read straight off the dataset arrays rather than through
    dataset[i], so `i` always means "row i" whatever --eval_stride the dataset
    was built with.

    The last block of a test period is almost never a whole pred_len long, but
    the model always consumes a full (label_len + pred_len) decoder window, so
    the missing rows are padded by repeating the last real one.  Nothing that is
    drawn depends on them: with --type_sampler dpm the sampler is conditioned on
    the input window only, and the prediction is cut back to the real number of
    remaining days in main().
    """
    seq_len, label_len, pred_len = dataset.seq_len, dataset.label_len, dataset.pred_len
    r_begin = i + seq_len - label_len
    r_end = r_begin + label_len + pred_len

    def pad(a, need):
        a = np.asarray(a)
        return a if len(a) >= need else np.concatenate(
            [a, np.repeat(a[-1:], need - len(a), axis=0)], axis=0)

    need = label_len + pred_len
    return (dataset.data_x[i: i + seq_len],
            pad(dataset.data_y[r_begin: r_end], need),
            dataset.data_stamp[i: i + seq_len],
            pad(dataset.data_stamp[r_begin: r_end], need),
            0,
            np.arange(i, i + seq_len),
            np.arange(r_begin, r_end),
            len(dataset.data_x))


def predict_windows(exp, dataset, window_ids, args, knob=None, chunk=None):
    """
    -> (n_windows, sample_times, pred_len, num_vars) log returns in the original scale

    knob = None reproduces the original two-condition model (the "before" curve);
    a float dials the extreme-scenario knob. Neither reads the future.

    The windows are pushed through the model in chunks of `chunk` (default
    --test_batch_size): a multi-year test range is 50+ windows and sampling them
    all at once does not fit on one GPU.
    """

    chunk = chunk or max(int(getattr(args, 'test_batch_size', 32) or 32), 1)
    exp.model.eval()

    outs = []
    for c in range(0, len(window_ids), chunk):
        batch = [window_item(dataset, i) for i in window_ids[c: c + chunk]]
        to_t = lambda k: torch.from_numpy(np.stack([b[k] for b in batch])).float().to(exp.device)

        batch_x, batch_y = to_t(0), to_t(1)
        batch_x_mark, batch_y_mark = to_t(2), to_t(3)

        with torch.no_grad():
            outputs, _, _, _, _ = exp.model.forward(
                batch_x, batch_x_mark, batch_y, batch_y_mark,
                sample_times=args.sample_times, knob=knob)

        out = outputs.detach().cpu().numpy()          # (B, S, pred_len, V)
        if out.ndim == 3:
            out = out[:, None]
        outs.append(out)

    out = np.concatenate(outs, axis=0)
    B, S, L, V = out.shape
    out = dataset.inverse_transform(out.reshape(-1, V)).reshape(B, S, L, V)
    return out


def gbm_returns(hist_ret, pred_len, popt):
    """
    Geometric Brownian motion fitted on one input window.
    hist_ret : (seq_len,) real log returns the model was given
    -> (gbm_paths, pred_len) simulated log returns
    """
    r = np.asarray(hist_ret, dtype=float)
    r = r[np.isfinite(r)]
    mu = float(r.mean()) if popt.gbm_drift == 'hist' else 0.0
    sigma = float(r.std(ddof=1))

    rng = np.random.default_rng(popt.gbm_seed)
    return mu + sigma * rng.standard_normal((popt.gbm_paths, pred_len))


KNOB_COLORS = ['#ff7f0e', '#9467bd', '#8c564b', '#17becf', '#7f7f7f', '#bcbd22']


def draw(hist_x, hist_y, segments, ylabel, title, out_path):
    """
    hist_x/hist_y : the real series before the first forecast
    segments      : list of dicts with x (dates incl. anchor), truth, pred, lo, hi
                    and optionally base / base_lo / base_hi for the GBM baseline
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.plot(hist_x, hist_y, '-', c='#000000', lw=1.2, label='History')

    for i, s in enumerate(segments):
        if s['lo'] is not None:
            ax.fill_between(s['x'], s['lo'], s['hi'], color='#f4b0b0', alpha=0.55, lw=0,
                            label='Prediction 10-90%' if i == 0 else None)
        if s.get('base_lo') is not None:
            ax.fill_between(s['x'], s['base_lo'], s['base_hi'], color='#2ca02c', alpha=0.15, lw=0,
                            label='GBM 10-90%' if i == 0 else None)
        ax.plot(s['x'], s['truth'], '-', c='b', lw=1.2,
                label='GroundTruth' if i == 0 else None)
        ax.plot(s['x'], s['pred'], '-', c='r', lw=1.4,
                label='Prediction' if i == 0 else None)
        for j, (k, y) in enumerate(s.get('knobs') or []):
            ax.plot(s['x'], y, '-', c=KNOB_COLORS[j % len(KNOB_COLORS)], lw=1.4,
                    label='Prediction (knob={:g})'.format(k) if i == 0 else None)
        if s.get('base_paths') is not None:
            for j, p in enumerate(s['base_paths']):
                ax.plot(s['x'], p, '-', c='#2ca02c', lw=0.6, alpha=0.35,
                        label='GBM sample paths' if i == 0 and j == 0 else None)
        if s.get('base') is not None:
            ax.plot(s['x'], s['base'], '-', c='#2ca02c', lw=1.0,
                    label='GBM baseline' if i == 0 else None)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


def main():
    popt = plot_args()
    args = get_args(known_only=True)

    setting = build_setting(args)
    ckpt_dir = os.path.join(args.checkpoints, setting)
    out_dir = popt.out_dir or os.path.join(ckpt_dir, 'price_plots')
    os.makedirs(out_dir, exist_ok=True)

    print('setting :', setting)
    print('mode    :', popt.plot_mode, '| sample_times =', args.sample_times)

    exp = Exp_Main(args)
    exp.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
    print('loaded  :', os.path.join(ckpt_dir, 'checkpoint.pth'))

    dataset, _ = data_provider(args, flag=popt.flag)

    cols = list(getattr(dataset, 'cols', []))
    close_cols = [c for c in cols if c.endswith('_r_close')]
    if close_cols:
        channels = [(c[:-len('_r_close')], cols.index(c), c) for c in close_cols]
    else:  # 1-feature-per-ticker file (column name == ticker)
        channels = [(c, i, c) for i, c in enumerate(cols)]
    print('channels:', [t for t, _, _ in channels])

    # real series, indexed by date
    ret = pd.read_csv(os.path.join(args.root_path, args.data_path), parse_dates=['date']).set_index('date')
    px = pd.read_csv(os.path.join(args.root_path, popt.price_file), parse_dates=['date']).set_index('date')

    seq_len, pred_len = args.seq_len, args.pred_len

    n_rows = len(dataset.data_x)

    if popt.plot_mode == 'window':
        window_ids = [popt.plot_window]
    else:
        # Every window that still has at least one day to forecast is kept.  The
        # last one is usually shorter than pred_len; it is drawn as a short block
        # rather than dropped, so the figure runs to the final day of the range.
        window_ids = list(range(0, n_rows - seq_len, pred_len))
    print('windows :', window_ids)

    dates = pd.to_datetime(dataset.dates)
    steps = {i: min(pred_len, n_rows - i - seq_len) for i in window_ids}
    tail = steps[window_ids[-1]]
    if tail < pred_len:
        print('tail    : last block is {} of {} steps ({} ~ {})'.format(
            tail, pred_len,
            dates[window_ids[-1] + seq_len].date(), dates[n_rows - 1].date()))

    preds = predict_windows(exp, dataset, window_ids, args, knob=None)      # (W, S, L, V)

    knob_values = [float(k) for k in popt.knob_values.split(',') if k.strip() != '']
    knob_preds = []
    for k in knob_values:
        print('knob    :', k)
        knob_preds.append((k, predict_windows(exp, dataset, window_ids, args, knob=k)))

    band = preds.shape[1] > 1 and bool(popt.plot_band)
    summary = []
    knob_summary = []

    for tic, v, ret_col in channels:

        first = window_ids[0]
        hist_days = popt.history_days if popt.history_days > 0 else seq_len
        hist_lo = max(first + seq_len - hist_days, 0)
        hist_dates = dates[hist_lo: first + seq_len]

        seg_ret, seg_px, errs, dir_hit = [], [], [], []
        gbm_errs, gbm_hit, chain_errs = [], [], []
        cum_ret = {'real': [], None: []}          # horizon cumulative log return per window
        for k, _ in knob_preds:
            cum_ret[k] = []

        # --price_chain: the generated price is one continuous path. Only the first
        # block is seeded with a real close; after that every block starts where the
        # previous generated block ended, so the line never snaps back to reality.
        chain = {'model': None, 'gbm': None}
        chain_r = {'model': None, 'gbm': None}      # same idea on the return figure
        for k, _ in knob_preds:
            chain[k] = None
            chain_r[k] = None

        for w, i in enumerate(window_ids):
            L = steps[i]                                 # < pred_len on the last block
            anchor_date = dates[i + seq_len - 1]
            in_dates = dates[i: i + seq_len]
            fut_dates = dates[i + seq_len: i + seq_len + L]
            xs = pd.DatetimeIndex([anchor_date]).append(pd.DatetimeIndex(fut_dates))

            r_pred = preds[w, :, :L, v]                      # (S, L) log returns
            r_true = ret[ret_col].reindex(fut_dates).values
            r_anchor = float(ret[ret_col].loc[anchor_date])

            # GBM fitted on the very window the model was given
            r_gbm = gbm_returns(ret[ret_col].reindex(in_dates).values, L, popt) \
                if popt.gbm else None

            # same model, same window, knob dialed to k  -> (S, pred_len)
            r_knob = [(k, kp[w, :, :L, v]) for k, kp in knob_preds]

            cum_ret['real'].append(float(np.nansum(r_true)))
            cum_ret[None].append(float(r_pred.mean(axis=0).sum()))
            for k, rk in r_knob:
                cum_ret[k].append(float(rk.mean(axis=0).sum()))

            # Every block is drawn starting at its anchor day so the segments touch.
            # Without chaining that point is the real return, which makes the
            # generated line jump back to reality at every block boundary; with
            # chaining it is the previous block's last generated return, i.e. the
            # same (date, value) the previous segment ended on -> one unbroken line.
            def ret_start_from(key):
                if not popt.price_chain or chain_r[key] is None:
                    return r_anchor
                return chain_r[key]

            r_pred_mean = r_pred.mean(axis=0)
            r_start = ret_start_from('model')
            r_start_gbm = ret_start_from('gbm')
            r_knob_lines = [(k, ret_start_from(k), rk.mean(axis=0)) for k, rk in r_knob]

            chain_r['model'] = float(r_pred_mean[-1])
            if popt.gbm:
                chain_r['gbm'] = float(r_gbm.mean(axis=0)[-1])
            for k, _, rk_mean in r_knob_lines:
                chain_r[k] = float(rk_mean[-1])

            seg_ret.append(dict(
                knobs=[(k, np.concatenate([[k_start], rk_mean])) for k, k_start, rk_mean in r_knob_lines],
                x=xs,
                truth=np.concatenate([[r_anchor], r_true]),
                pred=np.concatenate([[r_start], r_pred_mean]),
                lo=np.concatenate([[r_start], np.percentile(r_pred, 10, axis=0)]) if band else None,
                hi=np.concatenate([[r_start], np.percentile(r_pred, 90, axis=0)]) if band else None,
                base=np.concatenate([[r_start_gbm], r_gbm.mean(axis=0)]) if popt.gbm else None,
                base_paths=[np.concatenate([[r_start_gbm], p]) for p in r_gbm[:popt.gbm_show_paths]]
                    if popt.gbm and popt.gbm_show_paths else None,
                base_lo=np.concatenate([[r_start_gbm], np.percentile(r_gbm, 10, axis=0)]) if popt.gbm and popt.plot_band else None,
                base_hi=np.concatenate([[r_start_gbm], np.percentile(r_gbm, 90, axis=0)]) if popt.gbm and popt.plot_band else None))

            p_anchor = float(px[tic].loc[anchor_date])
            p_true = px[tic].reindex(fut_dates).values

            def start_from(key):
                """where this block's generated line begins"""
                if not popt.price_chain:
                    return p_anchor
                if chain[key] is None:
                    chain[key] = p_anchor        # seed: the last real close before the range
                return chain[key]

            p_start = start_from('model')
            paths = p_start * np.exp(np.cumsum(r_pred, axis=-1))   # (S, pred_len)
            p_mean = paths.mean(axis=0)
            chain['model'] = float(p_mean[-1])

            # the per-window numbers stay re-anchored on the real close, otherwise a
            # single early miss would dominate every window that follows it
            rescale = p_anchor / p_start
            p_mean_win = p_mean * rescale

            if popt.gbm:
                g_start = start_from('gbm')
                gbm_paths = g_start * np.exp(np.cumsum(r_gbm, axis=-1))   # (gbm_paths, pred_len)
                g_mean = gbm_paths.mean(axis=0)
                chain['gbm'] = float(g_mean[-1])
                g_mean_win = g_mean * (p_anchor / g_start)
                gbm_errs.append(np.mean(np.abs(g_mean_win - p_true) / p_true) * 100)
                gbm_hit.append(float(np.sign(g_mean_win[-1] - p_anchor) == np.sign(p_true[-1] - p_anchor)))

            px_knob = []
            for k, rk in r_knob:
                k_start = start_from(k)
                k_mean = (k_start * np.exp(np.cumsum(rk, axis=-1))).mean(axis=0)
                chain[k] = float(k_mean[-1])
                px_knob.append((k, np.concatenate([[k_start], k_mean])))

            seg_px.append(dict(
                knobs=px_knob,
                x=xs,
                truth=np.concatenate([[p_anchor], p_true]),
                pred=np.concatenate([[p_start], p_mean]),
                lo=np.concatenate([[p_start], np.percentile(paths, 10, axis=0)]) if band else None,
                hi=np.concatenate([[p_start], np.percentile(paths, 90, axis=0)]) if band else None,
                base=np.concatenate([[g_start], g_mean]) if popt.gbm else None,
                base_paths=[np.concatenate([[g_start], p]) for p in gbm_paths[:popt.gbm_show_paths]]
                    if popt.gbm and popt.gbm_show_paths else None,
                base_lo=np.concatenate([[g_start], np.percentile(gbm_paths, 10, axis=0)]) if popt.gbm and popt.plot_band else None,
                base_hi=np.concatenate([[g_start], np.percentile(gbm_paths, 90, axis=0)]) if popt.gbm and popt.plot_band else None))

            errs.append(np.mean(np.abs(p_mean_win - p_true) / p_true) * 100)
            chain_errs.append(np.mean(np.abs(p_mean - p_true) / p_true) * 100)
            dir_hit.append(float(np.sign(p_mean_win[-1] - p_anchor) == np.sign(p_true[-1] - p_anchor)))

        tag = '{}  |  {}-step forecast'.format(tic, pred_len)

        draw(hist_dates, ret[ret_col].reindex(hist_dates).values, seg_ret,
             'log return   log(C_t / C_t-1)', tag,
             os.path.join(out_dir, '{}_logret.png'.format(tic)))

        draw(hist_dates, px[tic].reindex(hist_dates).values, seg_px,
             'adjusted close (USD)', tag,
             os.path.join(out_dir, '{}_price.png'.format(tic)))

        summary.append((tic, float(np.mean(errs)), float(np.mean(chain_errs)),
                        100.0 * float(np.mean(dir_hit)),
                        float(np.mean(gbm_errs)) if popt.gbm else float('nan'),
                        100.0 * float(np.mean(gbm_hit)) if popt.gbm else float('nan')))
        knob_summary.append((tic, {k: float(np.mean(v)) for k, v in cum_ret.items()}))

    print()
    chain_col = '{:>12}'.format('chain MAPE%') if popt.price_chain else ''
    print('{:<7} {:>12}{} {:>16} {:>12} {:>16}'.format(
        'ticker', 'price MAPE%', chain_col, 'direction hit%', 'GBM MAPE%', 'GBM hit%'))
    for tic, mape, c_mape, hit, g_mape, g_hit in summary:
        print('{:<7} {:>12.2f}{} {:>16.1f} {:>12.2f} {:>16.1f}'.format(
            tic, mape, '{:>12.2f}'.format(c_mape) if popt.price_chain else '',
            hit, g_mape, g_hit))
    print('(direction hit% = sign of the {}-step cumulative return, over {} windows)'.format(
        pred_len, len(window_ids)))
    if tail < pred_len:
        print('(the last window is a short {}-step block and is included in every average above)'.format(tail))
    if popt.price_chain:
        print('(price MAPE% re-anchors every block on the real close; chain MAPE% is the '
              'continuous path, so its error compounds across blocks)')
    if popt.gbm:
        print('(GBM = geometric Brownian motion, {} drift, {} paths fitted per input window)'.format(
            popt.gbm_drift, popt.gbm_paths))

    if knob_values:
        # Does the knob actually move the forecast? The mean cumulative log return over
        # the horizon should decrease monotonically as the knob is turned down.
        keys = ['real', None] + knob_values
        head = lambda k: 'realized' if k == 'real' else ('no knob' if k is None else 'knob {:g}'.format(k))
        print()
        print('mean {}-day cumulative log return per window'.format(pred_len))
        print('{:<7}'.format('ticker') + ''.join('{:>12}'.format(head(k)) for k in keys))
        for tic, d in knob_summary:
            print('{:<7}'.format(tic) + ''.join('{:>12.4f}'.format(d[k]) for k in keys))
        if len(knob_values) > 1:
            desc = sorted(knob_values, reverse=True)
            bad = [tic for tic, d in knob_summary
                   if not all(d[desc[i]] >= d[desc[i + 1]] for i in range(len(desc) - 1))]
            print('(monotone in {}/{} tickers{})'.format(
                len(knob_summary) - len(bad), len(knob_summary),
                '' if not bad else '; not monotone: ' + ', '.join(bad)))


if __name__ == '__main__':
    main()
