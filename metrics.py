import torch

"x is the input/conditioning variable/s for the model"
"y is the output/target variable/s for the model"

def nll(x, y, model):
    """ negative log likelihood loss. """
    log_prob = -1 * model.log_prob(y_true=y, x=x).mean(0)
    return log_prob


def mse(x, y, model, pred=None):
    """ mean squared error. """
    if model is not None:
        pred = model(x)
    mse_loss = torch.nn.functional.mse_loss(pred, y)
    return mse_loss


def mae(x, y, model, pred=None):
    """ mean absolute error. """
    if model is not None:
        pred = model(x)
    MAELoss = torch.nn.L1Loss()
    mae_loss = MAELoss(pred, y)
    return mae_loss


def bce(x, y, model, pred=None):
    """ binary cross entropy. """
    BCELoss = torch.nn.BCELoss()
    if model is not None:
        pred = model(x)
    bce_loss = BCELoss(pred, y)
    return bce_loss
