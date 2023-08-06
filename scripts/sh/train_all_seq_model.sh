#!/bin/bash

echo 'gram - gru' && \
python scripts/fit_regression_models.py --dataset seq_gram --model gru --search gridsearch && \
echo 'mark - gru' && \
python scripts/fit_regression_models.py --dataset seq_mark --model gru --search gridsearch && \
echo 'scat 1 - gru' && \
python scripts/fit_regression_models.py --dataset seq_scat_1 --model gru --search gridsearch && \
echo 'scat 2 - gru' && \
python scripts/fit_regression_models.py --dataset seq_scat_2 --model gru --search gridsearch && \
echo 'gram - conv' && \
python scripts/fit_regression_models.py --dataset seq_gram --model conv --search gridsearch && \
echo 'mark - conv' && \
python scripts/fit_regression_models.py --dataset seq_mark --model conv --search gridsearch && \
echo 'scat 1 - conv' && \
python scripts/fit_regression_models.py --dataset seq_scat_1 --model conv --search gridsearch && \
echo 'scat 2 - conv' && \
python scripts/fit_regression_models.py --dataset seq_scat_2 --model conv --search gridsearch
