#!/bin/bash
python3 -m core.graph.trainer "dataset/dummy/train.csv" "dataset/dummy/dev.csv" "dataset/dummy/test.csv" 4 --lr=1e-3 --optimizer_name="Adam" --freeze_bert