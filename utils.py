import torch
from config import Config
import os.path as path
from ast import literal_eval

def load_model(model, checkpoint_path, device):
    """Load the saved models"""
    loaded_state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']
    model.load_state_dict(loaded_state_dict)
    return model


def save_run_params_in_file(folder_path, run_config):
    """
    Receives a config class, fetches all member variables and saves them
    in a config file for logging purposes.
    Parameters:
        folder_path - output folder
        run_config - shallow class with parameter members
    """
    with open(path.join(folder_path, f'run_params.conf'), 'w') as run_param_file:
        for attr, value in sorted(run_config.__dict__.items()):
            run_param_file.write(attr + ': ' + str(value) + '\n')


def getxy(batch):
    """This function returns the inputs/conditioning variables x and output/target variables y based on the Config
    features """
    X, Y = batch
    x = torch.stack([X[key] for key in Config.cond_features], dim=-1)
    y = torch.stack([Y[key] for key in Config.output_features], dim=-1)
    return x, y


def load_run_params_from_file(folder_path):
        with open(path.join(folder_path, 'run_params.conf'), 'r') as f:
            for line in f:
                (k, _, v) = line.partition(': ')
                if k.startswith('_') or k.startswith(' '):
                    continue
                k = k.strip()
                v = v.strip()
                try:
                    v = literal_eval(v)
                except ValueError:
                    pass
                if v is None:
                    continue
                setattr(Config, k, v)
