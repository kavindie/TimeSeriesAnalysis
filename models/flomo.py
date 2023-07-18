import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .rnn import RNN
from .spline_flow import NeuralSplineFlow


class FloMo(nn.Module):

    def __init__(self,
                 t_in,
                 t_out,
                 beta=0.2,
                 gamma=0.02,
                 n_layer=10,
                 f_cond=10,
                 f_mod=2,
                 device=0,
                 ):
        super().__init__()
        self.t_out = t_out
        self.output_size = self.t_out * f_mod
        self.f_mod = f_mod
        self.beta = beta
        self.gamma = gamma

        self.conditional_dim = 16
        self.obs_encoding_size = 16
        self.obs_encoder = RNN(nin=f_cond,
                               nout=self.obs_encoding_size,
                               device=device)

        self.flow = NeuralSplineFlow(nin=self.output_size,
                                     nc=self.conditional_dim,
                                     n_layers=n_layer,
                                     K=128,
                                     B=1.0,
                                     hidden_dim=[16, 16],
                                     device=device)

        # move model to specified device
        self.device = device
        self.to(self.device)

    def _encode_conditionals(self, x):
        # encode original observed data
        x_enc = self.obs_encoder(x)  # encode histories
        x_enc = x_enc[:, -1]
        return x_enc


    def _inverse(self, y_true, x):
        x_t = x[..., -1:, :]
        x_enc = self._encode_conditionals(x)  # history encoding
        y_rel_flat = torch.flatten(y_true, start_dim=1)

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2  # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)

            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)

        z, jacobian_det = self.flow.inverse(torch.flatten(y_rel_flat, start_dim=1), x_enc)
        return z, jacobian_det

    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError

    def sample(self, n, x):
        with torch.no_grad():
            x_enc = self._encode_conditionals(x)  # history encoding
            x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.conditional_dim)
            n_total = n * x.size(0)
            output_shape = (x.size(0), n, self.t_out, self.f_mod)  # predict n trajectories input

            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples, log_det = self.flow(z, x_enc_expanded)
            samples = samples.view(*output_shape)
            normal = Normal(0, 1/3, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), -1))

            return samples, log_probs

    def predict_expectation(self, n, x):
        samples = self.sample(n, x)
        y_pred = samples.mean(dim=1, keepdim=True)
        return y_pred

    def log_prob(self, y_true, x):
        z, log_abs_jacobian_det = self._inverse(y_true, x)
        normal = Normal(0, 1/3, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det
