import argparse

import torch

def parse_commandline():
    """
    Reads the inputs given at run time and assign them to the corresponding variable in Config.
    """
    parser = argparse.ArgumentParser(description='Run training motion prediction network.')
    parser.add_argument('--load', type=str, help='Load a model')
    parser.add_argument('--epochs', type=int,  help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,  help='Batch Size.')
    parser.add_argument('--csv_file', type=str,  help='quantTest_data.csv')
    parser.add_argument('--t_in', type=int, help='How many days are you observing?')
    parser.add_argument('--t_out', type=int, help='How many days are you predicting?')
    parser.add_argument('--f_cond', type=int,help='How many features are you conditioning?')
    parser.add_argument('--f_mod', type=int, help='How many features are you modelling?')
    parser.add_argument('--model_num', type=int, help='What model are you training?')
    parser.add_argument('--both_series', action='store_true', help='Are you training both series together?')
    parser.add_argument('--one_series', type=str, help='If training one series, what is it, ts1 or ts2?')

    args = parser.parse_args()
    for k, v in args._get_kwargs():
        if v is None:
            continue
        setattr(Config, k, v)

    # Only needs to be checked  and set if training. Otherwise, set from the saved config file
    if Config.load is None:
        # You cannot give true to training simultaneously and give a value to train seperately.
        assert Config.both_series or Config.one_series is not None
        # Set specific config params based on parse_commandline. Please refer to readme file.

        if Config.model_num == 4:
            Config.output_features_ = [
                'mean_binary'
            ]

        Config.cond_features = []
        Config.output_features = []
        if Config.both_series:
            s = ['ts1', 'ts2']
        else:
            s = [Config.one_series]

        for ts in s:
            for f in Config.cond_features_specific:
                Config.cond_features.append(f'{f}_{ts}')
        for ts in s:
            for f in Config.output_features_:
                Config.output_features.append(f'{f}_{ts}')

        Config.cond_features = [*Config.cond_features, *Config.cond_features_common]
        Config.f_cond = len(Config.cond_features)  # Conditioning feature_size
        Config.f_mod = len(Config.output_features)  # Modelling feature_size

        # Special config parameters if it is model number 3
        if Config.model_num == 3:
            Config.noise = True

            # model values
            Config.beta = 0.002
            Config.gamma = 0.0002
            if Config.both_series:
                Config.n_layer = 3
            else:
                Config.n_layer = 1


class Config:
    """
    A shallow class to save the training configurations
    """
    # training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # only necessary if you have GPU

    # Arguments passed
    epochs = 150  # Number of epochs to train for
    batch_size = 128
    csv_file = 'quantTest_data.csv'
    t_in = 7  # How many time steps of history?
    t_out = 1  # For how long are you predicting?
    load = None  # If you are laoding a model for evaluation

    '''We will be testing fro 4 models'''
    # Model 1: DNN
    # Model 2: RNN
    # Model 3: FloMo -proposed
    # Model 4: Binary classifer - not completed
    model_num = 3

    both_series = False  # Is it simultaneous series training?
    one_series = 'ts1'  # If it is just one series, what it is?

    cond_features_common = [
        'day_of_week_sin',
        'day_of_week_cos',
        'day_of_month_sin',
        'day_of_month_cos',
        'day_of_year_sin',
        'day_of_year_cos',
    ]  # What are the conditioning features common to both series?
    cond_features_specific = [
        'volume',
        'mean',
        'var',
        'mean_diff',
    ]   # What are the conditioning features specific to a series?
    output_features_ = [
        'mean_diff',
    ]   # What are the output features?

    sequence_length = t_in + t_out

    # To collect the relevant conditioning and modelling features. Will be set in parse_commandline()
    cond_features = []
    output_features = []
    f_cond = 0
    f_mod = 0

    # Special config parameters if it is model number 3
    noise = None
    beta = None
    gamma = None
    n_layer = None







