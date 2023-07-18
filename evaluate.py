import torch
from train import build_data, build_model, cuda_context
from config import parse_commandline, Config
from test_utils import TestUtils
from utils import load_run_params_from_file


def eval():
    # Function to evaluate your models
    # Loading the datasets and dataloader
    parse_commandline()
    load_dir = Config.load.rpartition('/')[0]

    # Loading the relevant config details
    load_run_params_from_file(load_dir)

    _, _, test_loader = build_data()

    # create raw model and loss func
    model, loss_func = build_model()

    # Loading the model
    model_name = Config.load.rpartition('/')[0].rpartition('/')[-1]
    print(f'Loading model {model_name}...')
    loaded_state_dict = torch.load(Config.load, map_location=Config.device)
    model.load_state_dict(loaded_state_dict)

    model_tester = TestUtils(model=model,
                             model_num=Config.model_num,
                             loss_func=loss_func,
                             test_loader=test_loader,
                             load_dir=load_dir,
                             both_series=Config.both_series,
                             one_series=Config.one_series,
                             )
    print(f"Starting fresh training...with model {model_name}")
    model_tester.start_testing()


if __name__ == "__main__":
    with cuda_context():
        eval()
