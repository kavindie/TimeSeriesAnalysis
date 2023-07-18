# Time Series Analysis

This project is for time series analysis, created for use in a trading system.

You can contact me at <hkkavindie@yahoo.com>.

## Originality

All the code herewith is written by me except for 

- `models/flomo.py`
- `models/rqs.py`
- `models/spline_flow.py`

written by Christoph Scholler, under an MIT licence. The original code of Christoph Scholler was modified to match the problem.

## Problem

You are given a `.csv` file (`quantTest_data.csv`) with two financial time series spanning 1-Jan-2008 to 1-Jan-2013.

The columns of the `.csv` file are of the form:
`time, ts1, ts2`.

Based on this make a predictive model on how the two time series vary over time independently and together.

## Solution

Each time series is separately modelled and predicted first.
Later, they are modelled jointly. 

Train, validate and test is split as 80%, 10% and 10%.

4 models are considered:
1. DNN
2. RNN
3. NF based FloMo
4. Binary classifier (not complete)

The past 7 days’ worth of data is input into each model. Below are the input features of each day:
1. Log mean of the daily trading
2. Difference in daily trading log mean
3. Variance of the daily trading log mean
4. Normalized volume of daily trending
5. Day of the Week (sin and cos values)
6. Day of the Month (sin and cos values)
7. Day of the Year (sin and cos values)

The models predict the next day’s log difference in daily trading mean (for models 1,2 and 3) or binary difference (model 4).


## Demonstration

### Dependencies
This project was developed in Python 3.6.
It uses packages
- `numpy` and
- `pandas` for data processing,
- `torch` for deep learning,
- `petname` for naming models,
- `statmodels` for baseline models, and
- `matplotlib` for visualisation.

You can install the exact versions used for development with the requirements file:
```shell
pip install -r requirements.txt
```

### Data
Please add the `quantTest_data.csv` dataset file in this folder.

### Training
The `train.py` script will train a model based on the provided data.
```shell
train.py --one_series ts1  # for time series 1 (the default)
train.py --one_series ts2  # for time series 2
train.py --both_series     # for the joint model
```
The `--model_num N` argument controls which model is trained, the following models are available N=:
1. DNN
2. RNN
3. FloMo NF model (default)
4. DNN binary classifier (not complete)

More command line arguments can be used to change the parameters of the model, see `train.py --help`.

Model weights and training statistics are saved in the
*Trained_Models/`ts1|ts2|both`/`model_num`/`model_name`/*
directory.


### Evaluation
The `evaluate.py` script will evaluate a previously trained model.
To evaluate the model, specify the path to the `model.pt` file.
```shell
evalute.py --load "Trained_models/ts1/3/$model_name/model.pt"  # Evaluates model 3 (FloMo) on time series 1
```
This will produce a `model_loss_summary.json` file and `prediction.png` in the model directory.

### Baseline Models
For comparison, a naive forecast model (shift by one day) and an ARIMA model are produced.
The `baseline.py` script will evaluate these baseline models in a manner comparible with `evaluate.py`.

## Other Important Files/Directories
- `models/` contains the model definitions.
- `metrics.py` defines loss functions and evaluation metrics.
- `Trained_Models/` contains saved models and evaluation results.
- `config.py` specifies the default configuration and script command line arguments.
- `dataset.py` performs data loading and cleaning.
- `quantTest_data.csv` dataset file should exist in this directory.
