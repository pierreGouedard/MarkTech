#!/bin/bash

python scripts/fit_regression_models.py --dataset full_gram --model xgb --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_mark --model xgb --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_scat_0 --model xgb --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_scat_1 --model xgb --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_scat_2 --model xgb --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_gram --model fc --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_mark --model fc --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_scat_0 --model fc --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_scat_1 --model fc --search gridsearch && \
python scripts/fit_regression_models.py --dataset full_scat_2 --model fc --search gridsearch