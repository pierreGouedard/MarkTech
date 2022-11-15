#!/bin/bash

echo 'lld - gru' && \
python scripts/train_models.py --dataset lld_seq --model gru && \
echo 'lld - tf' && \
python scripts/train_models.py --dataset lld_seq --model tf && \
echo 'vgg - tf' && \
python scripts/train_models.py --dataset vgg_seq --model tf && \
echo 'vgg - gru' && \
python scripts/train_models.py --dataset vgg_seq --model gru
