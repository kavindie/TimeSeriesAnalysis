import pandas as pd
import numpy as np
import torch
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import autocorrelation_plot
from dataset import csv2pandas

path = 'quantTest_data.csv'
df = csv2pandas(path)
test_len = 177  # Currently, this is manually filled to match the dataloader's output
series = 'ts1'  # TODO specify what series you are analysing
name = f'mean_diff_{series}'
train, target = df[:-test_len][name], df[-test_len:][name]


def naive():
    pred = df[-(test_len+1):-1][name]
    print_error(pred, target)
    plot_pred_target(pred, target)


def arima():
    final_aic = np.inf
    final_order = (0, 0, 0)
    for p in range(1, 4):
        for q in range(1, 4):
            for d in range(1, 2):
                res = ARIMA(train, order=(p, d, q)).fit()
                current_aic = res.aic
                if current_aic < final_aic:
                    final_aic = current_aic
                    final_order = (p, d, q)
    pred = []
    for t in range(len(target)):
        hist = pd.concat([train, target.iloc[:t]])
        model = ARIMA(hist, order=final_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        pred.append(yhat)

    print_error(pred, target)
    plot_pred_target(pred, target)


def print_error(pred, target):
    maeloss = torch.nn.L1Loss()
    print(f'MSE loss is {torch.nn.functional.mse_loss(torch.tensor(pred).float(), torch.tensor(target).float())}')
    print(f'MAE loss is {maeloss(torch.tensor(pred).float(), torch.tensor(target).float())}')


def plot_correlation_funcs():
    autocorrelation_plot(df[name])
    plt.show()
    plot_acf(df[name])
    plt.show()
    plot_pacf(df[name])
    plt.show()


def plot_pred_target(pred, target):
    plt.clf()
    plt.plot(np.asarray(target), '.-b')
    plt.plot(np.asarray(pred), '.-r')
    plt.legend(('Target', 'Prediction'))
    plt.show()


if __name__ == '__main__':
    naive()
    plot_correlation_funcs()
    arima()
