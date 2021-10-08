#!/bin/bash

dump_path="../dump_path"
data_path="../nli_dataset/Multi-genre-nli"

train_data=$data_path/multinli_1.0_train.csv
dev_data=$data_path/multinli_1.0_dev_matched.csv
test_data=$data_path/multinli_1.0_dev_mismatched.csv
batch_size=64
reload_checkpoint=None

python3 trainer.py \
		$train_data \
		$dev_data \
		$test_data \
		$batch_size \
		--model_name bert-base-uncased \
		--lr 0.00001 \
		--lr_decay 0.8 \
		--lr_patience_scheduling 3 \
		--max_length 512 \
		--max_epochs 10 \
		--val_check_interval 0.5 \
		--patience_early_stopping 5 \
		--validation_metrics val_acc \
		--hidden_dim 768 \
		--accumulate_grad_batches 1 \
		--train_n_samples -1 \
		--val_n_samples -1 \
		--test_n_samples -1 \
		--dropout 0.1 \
		--dpsa False \
		--in_memory True \
		--random_seed 2021 \
		--dump_path $dump_path \
		--exp_id text_entailment \
		--reload_checkpoint $reload_checkpoint \
		--eval_only False \
		--auto_scale_batch_size True \
		--auto_lr_find False \
		--deterministic False