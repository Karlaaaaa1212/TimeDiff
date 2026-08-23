
import os
import torch
from exp.exp_main import Exp_Main

from config import get_args

args = get_args()

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:

    for ii in range(args.itr):

        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dt{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len, 
            ii,
            args.stage_mode)

        if args.tag != '':
            setting += '_' + str(args.tag)

        if args.ablation_study_case != "none":
            setting += '_' + str(args.tag)

        exp = Exp(args)

        # ==============================
        # Pertraining 
        # ==============================
        if args.stage_mode == 'TWO':
            print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if args.model in ["MATS", "MATS2"]:
                # exp.mats_pretrain(setting)
                pass
            else:
                exp.pretrain(setting)
                pass

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        if args.model == "D3VAE":
            exp.D3VAE_train(setting)
        else:
            exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, mode="test")

        torch.cuda.empty_cache()
else:
    ii = args.ii

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dt{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            ii,
            args.stage_mode)

    if args.tag != '':
        setting += '_' + str(args.tag)

    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, mode="test")
    torch.cuda.empty_cache()

