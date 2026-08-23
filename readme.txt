$python main_ddpm.py --pretrain_epochs 10 --train_epochs 100 --is_training 1 --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 --ablation_study_F_type Linear  --cond_ddpm_num_layers 30 --ddpm_layers_II 10 --learning_rate 0.0001 --label_len 336 --use_window_normalization True

NOTE: --use_window_normalization now defaults to False; the ETTh1 numbers above were
reproduced with it ON, so pass --use_window_normalization True explicitly.
Whatever you train with, pass the SAME value at inference / plotting time.

(after train, is_training can be set to 0)

result saved in ./result_logs/result.txt
MSE loss for ETTh1 data univariate, multivariate reproduced


==========================================================================
Stock (Yahoo Finance daily log returns)
==========================================================================

$bash run_stock.sh                   # 63 / 21 / 21    (3m input -> 1m horizon)
$bash run_stock.sh 252 63 21 32      # seq_len label_len pred_len test_batch_size
   runs steps 1-5 below end to end; every artifact and the log are keyed by <setting>

1) download the data (auto_adjust=True -> 除權息/split adjusted close)

$python -m data_provider.download_stock_data
   default: AAPL AMGN CRM CSCO IBM INTC MSFT NKE VZ WMT, 2008-01-01 ~ 2024-12-31,
            3 features per ticker = 30 columns (4278 trading days)

     <TIC>_r_close = log(adj_close_t / adj_close_{t-1})    cross-day momentum
     <TIC>_r_high  = log(adj_high_t  / adj_close_t)        >= 0
     <TIC>_r_low   = log(adj_low_t   / adj_close_t)        <= 0

   --feature_set close  -> r_close only (10 columns, the earlier setup)
   output : ./datasets/prediction/stock/stock_logret.csv
            ./datasets/prediction/stock/stock_adjclose.csv (adjusted prices, used to rebuild price paths)

2) train + test  (split is by calendar date, not by ratio)

   The split puts the real crash regimes in TEST and keeps at least one crash in
   train, so the knob (step 2b) has extreme examples to learn from:

     train 2008-01-03 ~ 2017-12-29  (2516 days)  GFC 2008-09, 2011, 2015
     val   2018-01-02 ~ 2019-12-31  ( 502 days)  2018 Q4 selloff
     test  2020-01-02 ~ 2024-12-31  (1258 days)  COVID crash 2020-02/03, 2022 bear

   Chronological, so no look-ahead leakage; the scaler is fitted on train rows only.
   Score a single crisis episode by narrowing the test window (the loader still
   reaches seq_len rows back for the input history):
     --test_start 2020-02-19 --test_end 2020-03-23    COVID crash   (23 sessions)
     --test_start 2022-01-03 --test_end 2022-10-12    2022 bear market

$python main_ddpm.py --dataset_name stock --seq_len 63 --label_len 21 --pred_len 21 \
   --pretrain_epochs 10 --train_epochs 100 --is_training 1 \
   --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 \
   --ablation_study_F_type Linear --cond_ddpm_num_layers 30 --ddpm_layers_II 10 \
   --learning_rate 0.0001

   seq_len/label_len/pred_len are trading days (252 = 1 year, 21 = 1 month).
   The split dates can be changed with --train_start/--train_end/--val_start/--val_end/--test_start/--test_end.
   Univariate: --features S --target AAPL_r_close   Probabilistic: --sample_times 10 (gives CRPS + std band)

2b) the extreme-scenario knob (a third condition)

   The model already conditions on two things: the history-based future-mixup and the
   pretrained AR/dlinear initialization. The knob is a third one:

     knob_v = sum_t(future r_v) / sqrt(pred_len)      per channel, standardized space

   i.e. "how many sigma is the cumulative move over the horizon". While TRAINING it is
   read off the true future (teacher forcing) and hidden with probability
   --knob_p_uncond (once per sample, again per channel). At INFERENCE nothing is read
   from the future: you dial --knob yourself.

   Because of that dropout ONE set of weights gives both curves you want to compare:
     --knob not set  -> the original two-condition model  ("before")
     --knob -3       -> a 3-sigma crash over the horizon  ("after")

   --knob_channels close  dials only the *_r_close channels and leaves r_high/r_low
                          free (default); all = dial every channel
   --knob_guidance 2.0    classifier-free guidance, pushes the knob harder
   --use_knob 0           trains the plain original model (checkpoints of the two are
                          NOT interchangeable: the knob adds 2*num_vars conv channels)

   Reported test metrics always run with the null knob, so they stay leakage-free.

   CALIBRATION (63/21/21 run, 100 epochs). knob k asks for k*sqrt(pred_len) standardized
   units, i.e. for AAPL (sigma ~ 0.0195/day) knob -3 = a -27% move over 21 days. At
   --knob_guidance 1.0 the model only delivers about a sixth of that (-4%): the two
   original conditions dominate the third one. --knob_guidance 3.0 recovers most of it
   (-21%). So dial the knob for direction, dial the guidance for magnitude:

     COVID crash window (2020-02-19, realized AAPL -25.7% over the next 21 days)
       no knob                          +4.2%
       knob -3, guidance 1              -4.3%
       knob -3, guidance 3             -21.0%
       knob -10, guidance 1            -20.6%     (bigger knob works too, less cleanly)

3) plot the forecasts: two figures per ticker, each with History / GroundTruth / Prediction

   <TIC>_<mode>_logret.png   close log return  r = log(C_t / C_t-1)   (real units, not standardized)
   <TIC>_<mode>_price.png    price rebuilt from it:
                             P_hat[t+k] = P[t] * exp(sum_{j<=k} r_hat[t+j])
                             P[t] = real adj close on the last day of the input window

$python plot_price.py --dataset_name stock --seq_len 63 --label_len 21 --pred_len 21 \
   --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 \
   --ablation_study_F_type Linear --cond_ddpm_num_layers 30 --ddpm_layers_II 10 \
   --sample_times 20 --plot_mode year

   give it the SAME flags as training (it rebuilds args through config.get_args)
   --plot_mode year    : the whole test period, stitched from non-overlapping windows
   --plot_mode window  : a single window, pick it with --plot_window <i>
   --sample_times >1   : draws the 10-90% band of the diffusion samples
   output: ./checkpoints/<setting>/price_plots/

3b) knob comparison: the four lines (History / GroundTruth / no knob / knob)

$python plot_price.py <same flags> --plot_mode year --gbm 0 --knob_values=-3 \
   --out_dir ./checkpoints/<setting>/knob_plots

   --knob_values=-3,-1,0,1,3  overlays several knob settings and prints
                              "mean 21-day cumulative log return per window" for
                              realized / no-knob / each knob value. Turning the knob
                              down must lower that number monotonically -- that table
                              is the real test of whether the knob does anything;
                              the figure alone can be misleading.
   the knob lines are scenario generation, NOT forecasts: their MAPE is meaningless
   and is deliberately not reported.

   (the old per-window test<i>.png and MTS_errors*.png are off by default now;
    re-enable with --out_figures 1 / --vis_MTS_analysis 1)

4) check whether the loss really goes down

   during training:  ./checkpoints/<setting>/loss_curve.png  +  losses.csv  (per-epoch train/val)
   afterwards:       $python plot_losses.py                  # newest run
                     $python plot_losses.py --setting <setting> --print_every 5
