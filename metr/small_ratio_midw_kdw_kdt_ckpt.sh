#!/usr/bin/bash

datapath="$IN_PATH"
worldsize=8
bsz=256

base_keep_rate="$1"
midw="$2"
kdw="$3"
kdt="$4"
ckpt="$5"
note="$6"
now=$(date +"%Y%m%d_%H%M%S")
exp_name=metr/small-"$base_keep_rate"-midw"$midw"-kdw"$kdw"-kdt"$kdt"-ckpt"$note"_$now
logdir=./output/$exp_name

mkdir -p "$logdir"
echo "output dir: $logdir"

python3 -m torch.distributed.launch --nproc_per_node="$worldsize" --master_port="$((1112+RANDOM%20))" --use_env \
	exps/metr/main.py \
	--model deit_small_patch16_shrink_base \
	--fuse_token \
	--base_keep_rate "$base_keep_rate" \
	--input-size 224 \
	--sched cosine \
	--lr 2e-5 \
	--min-lr 2e-6 \
	--weight-decay 1e-6 \
	--batch-size "$bsz" \
	--shrink_start_epoch 0 \
	--warmup-epochs 0 \
	--shrink_epochs 0 \
	--epochs 30 \
	--dist-eval \
	--finetune $ckpt \
	--data-path $datapath \
	--output_dir $logdir \
	--mid_loss_weight "$midw" \
  --mid_kd_weight "$kdw" \
  --mid_kd_tau "$kdt" \
	2>&1 | tee -a "$logdir"/output.log


echo "output dir for the exp: $logdir"\
