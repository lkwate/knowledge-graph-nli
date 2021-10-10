#!/bin/bash
python3 -m core.graphNN "dataset/dummy/train.csv" "dataset/dummy/dev.csv" "dataset/dummy/test.csv" 1 --accumulate_grad_batches=4