#!/bin/bash
python3 -m core.graphNN "dataset/multinli_1.0/multinli_1.0_train.csv" "dataset/multinli_1.0/multinli_1.0_dev_matched.csv" 1 --accumulate_grad_batches=64