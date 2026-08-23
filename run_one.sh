#!/usr/bin/env bash
# One experiment, end to end: train -> test -> loss curve -> price plots -> knob plots
# -> stitched path.  Everything lands in checkpoints/<setting>/ and nothing else is
# touched, so two experiments never overwrite each other.
#
#   bash run_one.sh                                  # baseline (x_start), no tag
#   bash run_one.sh v --parameterization v           # tag "v", velocity target
#   bash run_one.sh vol --weight_pred_loss 0.1       # tag "vol", any extra flags
#   SEQ=252 LABEL=63 PRED=63 bash run_one.sh long    # a different window
#
# The first argument is the tag ("" or "-" means no tag); everything after it is passed
# straight to main_ddpm.py, plot_price.py and predict_path.py.
#
# Yahoo is NOT re-downloaded: download_stock_data.py has a fixed --end 2024-12-31, so a
# refetch cannot add data, only risk changing it.  Run it by hand if you need new rows.
set -e
cd "$(dirname "$0")"

SEQ=${SEQ:-63}; LABEL=${LABEL:-21}; PRED=${PRED:-21}
EPOCHS=${EPOCHS:-100}; PRE_EPOCHS=${PRE_EPOCHS:-10}
LR=${LR:-0.0001}; BS=${BS:-64}; TEST_BS=${TEST_BS:-32}
# 0 = skip the built-in test0/10/20/30 png+pkl dumps: they show 4 single windows
# of one channel, which volatility_check / distribution_check / price_plots all
# cover better.  OUT_FIGURES=1 brings them back.
OUT_FIGURES=${OUT_FIGURES:-0}
GPU=${GPU:-0}

TAG=${1:-}; [ "$TAG" = "-" ] && TAG=""
[ $# -gt 0 ] && shift
EXTRA="$@"

MF="--dataset_name stock --seq_len $SEQ --label_len $LABEL --pred_len $PRED \
    --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 \
    --ablation_study_F_type Linear --cond_ddpm_num_layers 30 --ddpm_layers_II 10"

SETTING=stock_${SEQ}_${PRED}_DDPM_stock_ftM_sl${SEQ}_ll${LABEL}_pl${PRED}_dt0_TWO
if [ -n "$TAG" ]; then
    MF="$MF --tag $TAG"
    SETTING=${SETTING}_${TAG}
fi
# plot_price.py / predict_path.py rebuild the model, so they need the same flags that
# change its shape or its parameterization -- not just the tag
MF="$MF $EXTRA"

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p result_logs
LOG=result_logs/${SETTING}.log
: > "$LOG"

run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
step() { echo; echo "===== $* =====" | tee -a "$LOG"; }

echo "setting : $SETTING"      | tee -a "$LOG"
echo "flags   : $MF"           | tee -a "$LOG"
echo "gpu     : $GPU"          | tee -a "$LOG"

step "[1/5] pretrain $PRE_EPOCHS + train $EPOCHS + test"
run python main_ddpm.py $MF --is_training 1 --out_figures $OUT_FIGURES \
    --pretrain_epochs $PRE_EPOCHS --train_epochs $EPOCHS \
    --learning_rate $LR --batch_size $BS --test_batch_size $TEST_BS

step "[2/5] loss curve"
run python plot_losses.py --setting $SETTING --print_every 10

step "[3/5] price + logret plots"
run python plot_price.py $MF --plot_mode year --plot_band 1 --test_batch_size 16

step "[4/5] knob comparison (knob=-3)"
run python plot_price.py $MF --plot_mode year --gbm 0 --knob_values=-3 \
    --test_batch_size 16 --out_dir ./checkpoints/$SETTING/knob_plots

step "[5/5] stitched path (stride $PRED, one continuous 2020-2024 line)"
run python predict_path.py $MF --test_batch_size 16

echo | tee -a "$LOG"
echo "done -> ./checkpoints/$SETTING/   (log: $LOG)" | tee -a "$LOG"
ls ./checkpoints/$SETTING | grep -vE '^test[0-9]+' | sed 's/^/  /' | tee -a "$LOG"
