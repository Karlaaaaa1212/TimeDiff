#!/usr/bin/env bash
# Full stock pipeline: download -> train -> loss curve -> price plots.
#
#   bash run_stock.sh                    # default 63 / 21 / 21 (3m input, 1m horizon)
#   bash run_stock.sh 252 63 21 19       # seq_len label_len pred_len test_bs (1y in, 1m out)
#   bash run_stock.sh 63 21 21 32 winnorm --use_window_normalization True
#                                  ^tag   ^extra flags forwarded to train + plots
#
# Everything a run produces (including its log) is keyed by <setting>, so two
# configurations never overwrite each other.
set -e

SEQ_LEN=${1:-63}
LABEL_LEN=${2:-21}
PRED_LEN=${3:-21}
TEST_BS=${4:-32}
TAG=${5:-}
if [ $# -gt 5 ]; then shift 5; EXTRA="$@"; else EXTRA=""; fi

MODEL_FLAGS="--dataset_name stock --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
        --ddpm_layers_I 5 --cond_ddpm_channels_conv 32 --ddpm_layers_inp 5 \
        --ablation_study_F_type Linear --cond_ddpm_num_layers 30 --ddpm_layers_II 10 $EXTRA"

SETTING=stock_${SEQ_LEN}_${PRED_LEN}_DDPM_stock_ftM_sl${SEQ_LEN}_ll${LABEL_LEN}_pl${PRED_LEN}_dt0_TWO
if [ -n "$TAG" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --tag $TAG"
    SETTING=${SETTING}_${TAG}
fi
LOG=result_logs/${SETTING}.log

mkdir -p result_logs
: > "$LOG"

run() {   # echo + run + tee, so the log always matches what landed on disk
    echo "$@" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

echo "setting: $SETTING" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "[1/5] download Yahoo Finance data -> log returns" | tee -a "$LOG"
run python -m data_provider.download_stock_data

echo | tee -a "$LOG"
echo "[2/5] pretrain 10 + train 100 epochs + test" | tee -a "$LOG"
run python main_ddpm.py $MODEL_FLAGS \
    --pretrain_epochs 10 --train_epochs 100 --is_training 1 \
    --learning_rate 0.0001 --batch_size 64 --test_batch_size $TEST_BS

echo | tee -a "$LOG"
echo "[3/5] loss curve" | tee -a "$LOG"
run python plot_losses.py --setting $SETTING --print_every 10

echo | tee -a "$LOG"
echo "[4/5] plots: log return + price, one pair per ticker (+ GBM baseline)" | tee -a "$LOG"
run python plot_price.py $MODEL_FLAGS --plot_mode year --plot_band 1

echo | tee -a "$LOG"
echo "[5/5] knob comparison: history / truth / no-knob / knob=-3" | tee -a "$LOG"
run python plot_price.py $MODEL_FLAGS --plot_mode year --gbm 0 \
    --knob_values=-3 --out_dir ./checkpoints/$SETTING/knob_plots

echo | tee -a "$LOG"
echo "done -> ./checkpoints/$SETTING/  (log: $LOG)" | tee -a "$LOG"
