"""
Shared CLI / config for TimeDiff.

main_ddpm.py and the plotting scripts all build their args from here, so a plot
script can be given exactly the same flags as the training run.
"""

import os
import argparse
import random
import torch
import numpy as np


def get_args(argv=None, known_only=False):

    # allow_abbrev=False: the plotting scripts parse their own flags alongside these,
    # and a prefix match (--stage silently becoming --stage_mode) writes results to the
    # wrong <setting> directory without any error.
    parser = argparse.ArgumentParser(allow_abbrev=False,
        description='Non-stationary Diffusion for Time Series Forecasting')

    # basic config
    parser.add_argument('--ii', type=int, default=0)
    parser.add_argument('--use_window_normalization', type=bool, default=False)

    parser.add_argument('--stage_mode', type=str, default="TWO", help="ONE, TWO")
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--out_figures', type=int, default=1)
    parser.add_argument('--vis_ar_part', type=int, default=0, help='status')
    parser.add_argument('--vis_MTS_analysis', type=int, default=1, help='status')
    parser.add_argument('--vis_history_len', type=int, default=336,
        help='how much input history test<i>.png draws; <=0 means the full seq_len. '
             'capped at seq_len either way')
    parser.add_argument('--vis_channel', type=int, default=0,
        help='which channel test<i>.png plots. stock col order is '
             '[0-9]=r_close, [10-19]=r_high, [20-29]=r_low; -1 = last column')

    parser.add_argument('--model', type=str, default='DDPM', 
        help='model name, options: [DDPM]')

    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='train epochs')

    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--beta_dist_alpha', type=float, default=-1)  # -1
    parser.add_argument('--our_ddpm_clip', type=float, default=100) # 100

    # data loader
    parser.add_argument('--seq_len', type=int, default=1440, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')
    parser.add_argument('--pred_len', type=int, default=168, help='prediction sequence length')


    parser.add_argument('--dataset_name', type=str, default='ETTh1')
    parser.add_argument('--weather_type', type=str, default='mintemp', help="['rain' 'mintemp' 'maxtemp' 'solar']")

    # stock (daily log returns from Yahoo Finance) -- date based split
    #
    # The split is chronological and picked so that the *real* crash regimes land in
    # test, while train still contains at least one crash for the knob to learn from:
    #   train 2008-2017 : GFC 2008-09, 2011 downgrade, 2015 devaluation
    #   val   2018-2019 : 2018 Q4 selloff
    #   test  2020-2024 : the COVID crash, the 2022 bear market and the recoveries after both
    # Nothing in train comes after the test period, so there is no look-ahead leakage;
    # the scaler is fitted on the train rows only (data_loader.Dataset_Stock).
    #
    # To score a single crisis episode, just narrow the test window, e.g.
    #   --test_start 2020-02-19 --test_end 2020-03-23   (COVID crash, 23 sessions)
    #   --test_start 2022-01-03 --test_end 2022-10-12   (2022 bear market)
    # the loader still reaches seq_len rows back for the input history.
    parser.add_argument('--stock_file', type=str, default='stock_logret.csv')
    parser.add_argument('--train_start', type=str, default='2008-01-01')
    parser.add_argument('--train_end', type=str, default='2017-12-29')
    parser.add_argument('--val_start', type=str, default='2018-01-02')
    parser.add_argument('--val_end', type=str, default='2019-12-31')
    parser.add_argument('--test_start', type=str, default='2020-01-02')
    parser.add_argument('--test_end', type=str, default='2024-12-31')
    # How far the evaluation window is moved between two val/test samples.
    #   1 (default) : the classic dense sweep -- one window per day, so with
    #                 pred_len=21 every day is forecast 21 times and the reported
    #                 metrics average heavily overlapping windows.
    #   0           : step by pred_len -- windows never overlap, every day is
    #                 forecast exactly once (63 days in -> 21 days out -> shift 21).
    #   n > 1       : step by n.
    # Training always stays dense: overlapping windows are what gives it samples.
    parser.add_argument('--eval_stride', type=int, default=1,
                        help='val/test window stride; 0 = pred_len (non-overlapping)')

    # ---- extreme-scenario knob -------------------------------------------------
    # A third condition next to the two the model already has (history-based
    # future-mixup + the pretrained AR/dlinear initialization).
    #
    #   knob_v = sum_t(future r_v) / sqrt(pred_len)   in the standardized space,
    #
    # i.e. "how many sigma is the cumulative move over the horizon", one scalar per
    # channel. During training it is read off the *true* future (teacher forcing) and
    # dropped with probability knob_p_uncond, so the same weights also work with no
    # knob at all. At inference nothing is read from the future: you dial --knob
    # yourself, and the reported test metrics use the null knob.
    parser.add_argument('--use_knob', type=int, default=1,
        help='0 = the original two-condition model (checkpoints are not interchangeable)')
    parser.add_argument('--knob_type', type=str, default='cumret', choices=['cumret'],
        help='cumret: pred_len-day cumulative log return / sqrt(pred_len)')
    parser.add_argument('--knob_p_uncond', type=float, default=0.15,
        help='classifier-free dropout: prob of hiding the knob (applied once per sample '
             'and again per channel), so the model can also run unconditionally')
    parser.add_argument('--knob', type=float, default=None,
        help='inference-time knob value in sigma units, e.g. -3 = a 3-sigma crash over '
             'the horizon. None = no knob (the "before" curve)')
    parser.add_argument('--knob_channels', type=str, default='close', choices=['close', 'all'],
        help='close: only dial the *_r_close channels (r_high/r_low stay unconditioned) | '
             'all: dial every channel')
    parser.add_argument('--knob_guidance', type=float, default=1.0,
        help='classifier-free guidance scale at sampling time; 1.0 = plain conditioning')

    # Transformer datasets: ECL,ETTh1,ETTh2,ETTm1,ETTm2,Exchange,traffic,weather,illnes,wind

    # Monash datasets:  https://zenodo.org/communities/forecasting/search?page=3&size=20#
    # > weather_dataset: 1332/65981 (3010, 65981)
    # > sunspot_dataset_without_missing_values: (1, 73924)
    # > [half_hourly] elecdemand_dataset: (Electricity Demand (Elecdemand) Dataset) (1, 17520)
    # > [daily] saugeenday_dataset (https://zenodo.org/record/4656058#.Y4cTZWhByUk) (1, 23741)
    # > wind_4_seconds_dataset: (1, 7397147)
    #[not good] > dominick_dataset: 28/393  (115704, 393)
    #[not good] > covid_deaths_dataset: (266-num_vars, 212-seq_len)

    # depts datasets
    # > caiso
    # > production
    # > caiso_m
    # > production_m
    # > synthetic
    # > system_KS

    # Following are for regression. doesnt work in this version.
    # "AustraliaRainfall","HouseholdPowerConsumption1","HouseholdPowerConsumption2","BeijingPM25Quality","BeijingPM10Quality","Covid3Month","LiveFuelMoistureContent","FloodModeling1","FloodModeling2","FloodModeling3","AppliancesEnergy","BenzeneConcentration","NewsHeadlineSentiment","NewsTitleSentiment","BIDMC32RR","BIDMC32HR","BIDMC32SpO2","IEEEPPG","PPGDalia"

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--num_vars', type=int, default=7, help='encoder input size')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # Diffusion Models
    parser.add_argument('--interval', type=int, default=1000, help='number of diffusion steps')
    parser.add_argument('--ot-ode', default=True, help='use OT-ODE model')
    parser.add_argument("--beta-max", type=float, default=0.3, help="max diffusion for the diffusion model")
    parser.add_argument("--t0", type=float, default=1e-4, help="sigma start time in network parametrization")
    parser.add_argument("--T", type=float, default=1., help="sigma end time in network parametrization")
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--nfe', type=int, default=100)
    parser.add_argument('--dim_LSTM', type=int, default=64)

    parser.add_argument('--diff_steps', type=int, default=100, help='number of diffusion steps')
    parser.add_argument('--UNet_Type', type=str, default='CNN', help=['CNN'])
    parser.add_argument('--D3PM_kernel_size', type=int, default=5)
    parser.add_argument('--use_freq_enhance', type=int, default=0)
    parser.add_argument('--type_sampler', type=str, default='dpm', help=["none", "dpm"])
    parser.add_argument('--parameterization', type=str, default='x_start',
        choices=['x_start', 'noise', 'v'],
        help='what the network predicts. x_start=x0 (samples collapse: the net can ignore '
             'x_t and decode cond), noise=eps (samples explode: rebuilding x0 divides by '
             'sqrt(alpha_bar)=2.4e-7), v=velocity (recommended).')

    parser.add_argument('--ddpm_inp_embed', type=int, default=256)
    parser.add_argument('--ddpm_dim_diff_steps', type=int, default=256)
    parser.add_argument('--ddpm_channels_conv', type=int, default=256)
    parser.add_argument('--ddpm_channels_fusion_I', type=int, default=256)
    parser.add_argument('--ddpm_layers_inp', type=int, default=5)
    parser.add_argument('--ddpm_layers_I', type=int, default=5)
    parser.add_argument('--ddpm_layers_II', type=int, default=5)
    parser.add_argument('--cond_ddpm_num_layers', type=int, default=5)
    parser.add_argument('--cond_ddpm_channels_conv', type=int, default=64)

    parser.add_argument('--ablation_study_case', type=str, default="none", help="none, mix_1, ar_1, mix_ar_0, w_pred_loss")
    parser.add_argument('--weight_pred_loss', type=float, default=0.0)
    parser.add_argument('--ablation_study_F_type', type=str, default="CNN", help="Linear, CNN")
    parser.add_argument('--ablation_study_masking_type', type=str, default="none", help="none, hard, segment")
    parser.add_argument('--ablation_study_masking_tau', type=float, default=0.9)

    # forecasting task

    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--batch_size', type=int, default=64, help='32 batch size of train input data')  # 32
    parser.add_argument('--test_batch_size', type=int, default=32, help='32 batch size of train input data')  # 32

    # parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--tag', type=str, default='')


    if known_only:
        args, _ = parser.parse_known_args(argv)
    else:
        args = parser.parse_args(argv)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    args.DATAdir = "./datasets"
    args.data = "custom"
    if args.dataset_name == "ECL":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/electricity/')
        args.data_path = 'electricity.csv'
        args.use_valset = True
    if args.dataset_name == "ETTh1":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
        args.data_path = 'ETTh1.csv'
        args.data = "ETTh1"
        args.use_valset = True
    if args.dataset_name == "ETTh2":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
        args.data_path = 'ETTh2.csv'
        args.data = "ETTh2"
        args.use_valset = True
    if args.dataset_name == "ETTm1":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
        args.data_path = 'ETTm1.csv'
        args.data = "ETTm1"
        args.use_valset = True
    if args.dataset_name == "ETTm2":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/ETT-small/')
        args.data_path = 'ETTm2.csv'
        args.data = "ETTm2"
        args.use_valset = True
    if args.dataset_name == "Exchange":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/exchange_rate/')
        args.data_path = 'exchange_rate.csv'
        args.use_valset = True
    if args.dataset_name == "traffic":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/traffic/')
        args.data_path = 'traffic.csv'
        args.use_valset = True
    if args.dataset_name == "weather":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/weather/')
        args.data_path = 'weather.csv'
        args.use_valset = True
    if args.dataset_name == "wind":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/wind/')
        args.data_path = 'wind.csv'
        args.use_valset = True
        args.data = "wind"
        args.target = 'wind_power'
    if args.dataset_name == "stock":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/stock/')
        args.data_path = args.stock_file
        args.data = "stock"
        args.freq = 'b'          # business day -> [DayOfWeek, DayOfMonth, DayOfYear]
        args.use_valset = True

        import pandas as _pd
        _cols = [c for c in _pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=1).columns
                 if c != 'date']
        if args.target not in _cols:
            args.target = _cols[-1]
        args.num_vars = 1 if args.features == 'S' else len(_cols)
        print("[stock] tickers: {} | features={} | num_vars={} | target={}".format(
            _cols, args.features, args.num_vars, args.target))

        # Dataset_Stock moves `target` to the last column; mirror that here so the
        # knob channel indices line up with the tensors the model actually sees.
        _ordered = [c for c in _cols if c != args.target] + [args.target]
        args.knob_close_idx = [i for i, c in enumerate(_ordered) if c.endswith('_r_close')]
        if not args.knob_close_idx:      # 1-feature-per-ticker file: every column is a close
            args.knob_close_idx = list(range(len(_ordered)))

    if args.dataset_name == "illness":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/illness/')
        args.data_path = 'national_illness.csv'
        args.use_valset = True

    if args.dataset_name in ["covid_deaths_dataset","sunspot_dataset_without_missing_values","elecdemand_dataset","saugeenday_dataset","wind_4_seconds_dataset","dominick_dataset","weather_dataset"]:
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_monash')
        args.data_path = ''
        args.use_valset = True

    if args.dataset_name in ["caiso", "caiso_m"]:
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/caiso/')
        args.data_path = ''
        args.data = args.dataset_name
        args.use_valset = True
    if args.dataset_name in ["production", "production_m"]:
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/nordpool/')
        args.data_path = ''
        args.data = args.dataset_name
        args.use_valset = True
    if args.dataset_name == "synthetic":
        args.synthetic_mode = 'L'  # ['L', 'Q', 'C', 'LT', 'QT', 'CT']
        args.model_id = "{}_{}_{}_{}".format(args.dataset_name, args.synthetic_mode, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/synthetic/')
        args.data_path = ''
        args.data = 'synthetic'
        args.use_valset = True
    if args.dataset_name == "system_KS":
        args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
        args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/dynamic_KS/')
        args.data_path = ''
        args.data = 'system_KS'
        args.use_valset = True


    return args
