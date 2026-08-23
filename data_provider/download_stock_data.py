"""
Download dividend/split adjusted (除權息調整後) daily OHLC from Yahoo Finance and
turn it into log-return features for TimeDiff.

Three features per ticker (feature_set='chl', the default -> 3 x n_tickers columns):

    <TIC>_r_close = log(adj_close_t / adj_close_{t-1})   cross-day momentum
    <TIC>_r_high  = log(adj_high_t  / adj_close_t)       >= 0, intraday upside range
    <TIC>_r_low   = log(adj_low_t   / adj_close_t)       <= 0, intraday downside range

feature_set='close' keeps only the r_close columns (the old 1-feature-per-ticker file).

Usage:
    python -m data_provider.download_stock_data
    python -m data_provider.download_stock_data --feature_set close
    python -m data_provider.download_stock_data --tickers AAPL MSFT --start 2008-01-01

Output: ./datasets/prediction/stock/stock_logret.csv   (date + the feature columns)
        ./datasets/prediction/stock/stock_adjclose.csv (adjusted close, to rebuild prices)
"""

import os
import argparse

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT"]


def download_log_returns(tickers, start, end, feature_set='chl'):
    """auto_adjust=True -> Close/High/Low are already adjusted for splits and dividends."""

    # yfinance treats `end` as exclusive, so push it one day forward
    end_exclusive = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw is None or len(raw) == 0:
        raise RuntimeError("Yahoo Finance returned no data. Check the tickers / date range.")

    def field(name):
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[name]
        else:  # single ticker
            df = raw[[name]].rename(columns={name: tickers[0]})
        return df[tickers]

    adj_close, adj_high, adj_low = field("Close"), field("High"), field("Low")

    # keep only days where every ticker has a full quote
    ok = adj_close.notna().all(axis=1) & adj_high.notna().all(axis=1) & adj_low.notna().all(axis=1)
    adj_close, adj_high, adj_low = adj_close[ok], adj_high[ok], adj_low[ok]

    r_close = np.log(adj_close / adj_close.shift(1))
    r_high = np.log(adj_high / adj_close)
    r_low = np.log(adj_low / adj_close)

    feats = {}
    for tic in tickers:
        feats['{}_r_close'.format(tic)] = r_close[tic]
    if feature_set == 'chl':
        for tic in tickers:
            feats['{}_r_high'.format(tic)] = r_high[tic]
        for tic in tickers:
            feats['{}_r_low'.format(tic)] = r_low[tic]

    out = pd.DataFrame(feats)
    out = out.dropna(how="any")   # drops the first row (r_close needs a previous close)

    return adj_close, out


def main():
    parser = argparse.ArgumentParser(description="Yahoo Finance -> log-return features for TimeDiff")
    parser.add_argument("--tickers", type=str, nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", type=str, default="2008-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--feature_set", type=str, default="chl", choices=["chl", "close"],
                        help="chl: r_close + r_high + r_low (3 per ticker) | close: r_close only")
    parser.add_argument("--out_dir", type=str, default="./datasets/prediction/stock/")
    parser.add_argument("--out_name", type=str, default="stock_logret.csv")
    parser.add_argument("--save_prices", type=int, default=1,
                        help="also save the adjusted close prices (needed to turn returns back into prices)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    adj_close, feats = download_log_returns(args.tickers, args.start, args.end, args.feature_set)

    df = feats.reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.columns.name = None

    out_path = os.path.join(args.out_dir, args.out_name)
    df.to_csv(out_path, index=False)

    if args.save_prices:
        px = adj_close.reset_index()
        px = px.rename(columns={px.columns[0]: "date"})
        px["date"] = pd.to_datetime(px["date"]).dt.strftime("%Y-%m-%d")
        px.columns.name = None
        px.to_csv(os.path.join(args.out_dir, "stock_adjclose.csv"), index=False)

    print("saved:", out_path)
    print("rows :", len(df), " range:", df["date"].iloc[0], "->", df["date"].iloc[-1])
    print("cols : {} features ({} tickers x {})".format(
        len(df.columns) - 1, len(args.tickers), 3 if args.feature_set == 'chl' else 1))
    print()
    print(feats.describe().T[["mean", "std", "min", "max"]])
    print()
    # mirrors the default split in config.py: the crash regimes land in test
    for name, (s, e) in {
        "train": ("2008-01-01", "2017-12-29"),
        "val":   ("2018-01-02", "2019-12-31"),
        "test":  ("2020-01-02", "2024-12-31"),
    }.items():
        n = ((df["date"] >= s) & (df["date"] <= e)).sum()
        print("  {:5s} {} ~ {} : {:5d} trading days".format(name, s, e, n))


if __name__ == "__main__":
    main()
