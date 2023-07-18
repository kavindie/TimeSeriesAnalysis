import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from utils import getxy
import torch
from metrics import mse, mae
import json


class TestUtils:
    def __init__(self,
                 model,
                 model_num,
                 loss_func,
                 test_loader,
                 load_dir,
                 both_series=False,
                 one_series='ts1',
                 ):
        """
        This class handles the inference of a pytorch model.
        :param model (nn.Module): model to be trained
        :param model_num (int): number specifying the model being trained
        :param loss_func (function): loss function that has the following parameters (input x, target y, model, pred).
        Target is a must. Either model or pred has to be given. If model is given, x must be given.
        :param test_loader (utils.data.DataLoader): testing data
        :param load_dir (path):  path to load the best saved model
        :param both_series (bool): Are both series used simultaneously?
        :param one_series (str) 'ts1' or 'ts2': If we are evaluating just one series, what is it?
        """
        self.model = model
        self.model_num = model_num
        self.loss = loss_func
        self.test_loader = test_loader
        self.load_dir = load_dir
        self.both_series = both_series
        self.one_series = one_series

        self.ys = []  # Will collect the groud truth mean log diff and mean log values for each iteration
        self.preds = []  # Will collect the predicted mean log diff and mean log values for each iteration
        self.metrics = defaultdict(list)  # Collect metrics for comparison. Will collect mse, mae and number of params

    def start_testing(self):
        self.model.eval()  # validate mode
        with torch.no_grad():
            predictions = []
            for step, batch in enumerate(self.test_loader, 1):
                x, y = getxy(batch)
                if self.model_num == 3:
                    pred, loss, pred_logprobs, y_logprob = self.test_step_3(x, y)
                    predictions.append(pred)
                    # If it is model 3, sample n and plot the pdf
                    if self.model_num == 3:
                        if step < 5: # Only plotting the first 5. This can be changed if you want more plots
                            self.plot_pdf(100, x, pred)

                    pred = pred[:, pred_logprobs.argmax(), ...]  # Likeliest sample
                    # pred = (pred*(pred_logprobs.exp())).mean()  # Expected sample
                else:
                    pred, loss = self.test_step(x, y)

                print(f'Test loss is {loss}')

                if self.both_series:
                    pred_mean_1 = batch[0]['mean_ts1'][0, -1] + pred[0, 0, 0]  # Get the mean log value of ts1 based on the predicted differnce
                    pred_mean_2 = batch[0]['mean_ts2'][0, -1] + pred[0, 0, 1]  # Get the mean log value of ts2 based on the predicted differnce
                    pred_mean = [pred_mean_1, pred_mean_2]
                    self.ys.append((*y, *[batch[1][f'mean_{s}'] for s in ['ts1', 'ts2']]))
                    self.preds.append((*pred, *pred_mean))
                    self.calculate_metrics(x, y[..., 0], pred[..., 0], series='ts1')
                    self.calculate_metrics(x, y[..., 1], pred[..., 1], series='ts2')
                else:
                    pred_mean = batch[0][f'mean_{self.one_series}'][0, -1] + pred
                    self.ys.append((y, batch[1][f'mean_{self.one_series}']))
                    self.preds.append((pred, pred_mean))
                    self.calculate_metrics(x, y, pred)

        params = sum(p.numel() for p in self.model.parameters())
        self.metrics['params'].append(params)

        self.write_summary()  # Save the metrics
        self.plot_losses()  # Plotting the ts1 and ts2 loss

    def test_step(self, x, y):
        try:
            # evaluate loss
            pred = self.model(x)
            if self.model_num == 2:
                pred = pred[..., -1, :][..., None, :]  # Get the last step of GRU
            loss = self.loss(x, y, model=None, pred=pred)
        except Exception as e:
            print(e, file=sys.stderr)
        return pred, loss

    def test_step_3(self, x, y):
        try:
            sample_val = 100
            # evaluate loss
            loss = self.loss(x, y, self.model)
            y_logprob = self.model.log_prob(y, x)
            pred, pred_logprobs = self.model.sample(sample_val, x)
        except Exception as e:
            print(e, file=sys.stderr)
        return pred, loss, pred_logprobs, y_logprob

    def plot_losses(self):
        if self.both_series:
            ts1_t, ts1_p = [], []
            ts2_t, ts2_p = [], []
            for i in range(len(self.ys)):
                ts1_t.append(self.ys[i][1])
                ts2_t.append(self.ys[i][2])
                ts1_p.append(self.preds[i][1])
                ts2_p.append(self.preds[i][2])
            plt.plot(np.asarray(ts1_t), '.-b')
            plt.plot(np.asarray(ts1_p), '.-r')
            plt.savefig(f'{self.load_dir}/prediction_ts1.png')
            plt.close()

            plt.plot(np.asarray(ts2_t), '.-b')
            plt.plot(np.asarray(ts2_p), '.-r')
            plt.savefig(f'{self.load_dir}/prediction_ts2.png')
            plt.close()
        else:
            plt.plot(np.asarray(self.ys)[:, 1], '.-b')
            plt.plot(np.asarray(self.preds)[:, 1], '.-r')
            plt.savefig(f'{self.load_dir}/prediction.png')
            plt.close()

    def plot_pdf(self, n, x, pred):
        if self.both_series:
            # grid = torch.stack(torch.meshgrid(
            #     torch.linspace(pred[0, :, 0, 0].min(), pred[0, :, 0, 0].max()),
            #     torch.linspace(pred[0, :, 0, 1].min(), pred[0, :, 0, 1].max()),
            # ), dim=-1)
            # Zoomed in grid
            grid = torch.stack(torch.meshgrid(
                torch.linspace(-0.01, 0.01),
                torch.linspace(-0.01, 0.01),
            ), dim=-1)
            logprobs = self.model.log_prob(grid.reshape(n * n, -1), x.expand(n * n, -1, -1))
            # plt.imshow(logprobs.reshape(n, n))
            # plt.title('logprobs')
            # plt.show()
            probs = logprobs.exp()
            plt.imshow(probs.reshape(n, n))
            plt.title('probabilities of the samples')
            plt.show()
        else:
            y_linspace = torch.linspace(pred.min(), pred.max(), n)
            y_prob = self.model.log_prob(y_linspace.reshape(n, 1, 1), x.expand(n, -1, -1)).exp()
            plt.plot(x[0, :, 1], '.-')
            plt.plot(7 + torch.zeros_like(pred.squeeze()), x[0, -1, 1] + pred.squeeze(), '_')
            plt.plot(7 + y_prob / y_prob.max(), x[0, -1, 1] + y_linspace)
            plt.show()

    def calculate_metrics(self, x, y, pred, series=None):
        mse_loss = mse(x, y, None, pred)
        mae_loss = mae(x, y, None, pred)

        if series is None:
            self.metrics['mse'].append(mse_loss.item())
            self.metrics['mae'].append(mae_loss.item())
        else:
            self.metrics[f'mse_{series}'].append(mse_loss.item())
            self.metrics[f'mae_{series}'].append(mae_loss.item())

    def write_summary(self):
        for key in list(self.metrics):
            self.metrics[f'{key}_mean'] = torch.as_tensor(self.metrics[key], dtype=float).mean()

        for key in list(self.metrics):
            self.metrics[key] = torch.tensor(self.metrics[key]).tolist()

        a_file = open(f'{self.load_dir}/model_loss_summary.json', "w")
        json.dump(self.metrics, a_file)
        a_file.close()
