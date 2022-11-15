#!/bin/bash

python scripts/train_models.py --dataset lld_full --model xgb && \
python scripts/train_models.py --dataset lld_full --model tf && \
python scripts/train_models.py --dataset vgg_full --model xgb && \
python scripts/train_models.py --dataset vgg_full --model tf
