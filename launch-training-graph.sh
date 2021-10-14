#!/bin/bash
python3 -m core.graphNN "dataset/dummy/train.csv" "dataset/dummy/dev.csv" "dataset/dummy/test.csv" 1 --accumulate_grad_batches=4 --lr=1e-3 --optimizer_name="Lamb"