import petname
import os
import torch.nn
from contextlib import contextmanager
from torch.utils.data import DataLoader, Subset


from dataset import FinanceDataset, SeqDataset
from config import Config, parse_commandline
import metrics
from train_utils import TrainUtils
from utils import save_run_params_in_file
from models.rnn import RNN
from models.dnn import DNN, DNNClassifier
from models.flomo import FloMo


def build_model():
    """
    Builds the model relevant for the model number and pick the corresponding loss function.
    """
    model = None
    loss_func = None
    # Model 1: DNN
    if Config.model_num == 1:
        model = DNN(
            t_in=Config.t_in,
            f_cond=Config.f_cond,
            t_out=Config.t_out,
            f_mod=Config.f_mod,
            device=Config.device,
        )
        loss_func = metrics.mse
    # Model 2: RNN
    elif Config.model_num == 2:
        model = RNN(
            nin=Config.f_cond,
            nout=Config.f_mod,
            es=16,
            hs=16,
            nl=3,
            device=Config.device,
        )
        loss_func = metrics.mse
    # Model 3: FloMo (Proposed)
    elif Config.model_num == 3:
        # Based on FloMo Tractable motion prediction with normalizing flows
        model = FloMo(Config.t_in,
                      Config.t_out,
                      beta=Config.beta,
                      gamma=Config.gamma,
                      f_cond=Config.f_cond,
                      f_mod=Config.f_mod,
                      device=Config.device,
                      n_layer=Config.n_layer,
                      )
        loss_func = metrics.nll
    # Model 4: Classifier - Not complete
    elif Config.model_num == 4:
        model = DNNClassifier(
            t_in=Config.t_in,
            f_cond=Config.f_cond,
            t_out=Config.t_out,
            f_mod=Config.f_mod,
            device=Config.device,
        )
        loss_func = metrics.bce
    else:
        raise NotImplementedError(f'model_num = {Config.model_num} not implemented')
    return model, loss_func


def build_data():
    """
    Generates the train, validate and test dataloaders as per the .csv file
    """
    dataset = FinanceDataset(Config.csv_file, Config.device)
    train_len = int(len(dataset) * 0.8)  # Breaking the data into 80% train
    val_len = int(len(dataset) * 0.1)  # 10% validate
    test_len = len(dataset) - (train_len + val_len)  # 10% test

    train_dataset = Subset(dataset, range(0, train_len))
    val_dataset = Subset(dataset, range(train_len, train_len + val_len))
    test_dataset = Subset(dataset, range(train_len + val_len, train_len + val_len + test_len))

    train_loader = DataLoader(SeqDataset(train_dataset,
                                         input_len=Config.t_in,
                                         target_len=Config.t_out,
                                         step=1),
                              batch_size=Config.batch_size,
                              shuffle=True)
    val_loader = DataLoader(SeqDataset(val_dataset,
                                       input_len=Config.t_in,
                                       target_len=Config.t_out,
                                       step=1),
                            batch_size=Config.batch_size,
                            shuffle=False)
    test_loader = DataLoader(SeqDataset(test_dataset,
                                        input_len=Config.t_in,
                                        target_len=Config.t_out,
                                        step=1),
                             batch_size=1,
                             shuffle=True)

    print(f'Training has {len(train_loader) * Config.batch_size} entries')
    print(f'Validation has  {len(val_loader) * Config.batch_size} entries')
    print(f'Testing has {len(test_loader)} entries')
    return train_loader, val_loader, test_loader


def main():
    """
    Loads the dataloaders, model and start training.
    Saves important details of the training.
    Relevant input argumnets need to be passed in as part of the running command.
    """
    # Loading the datasets and dataloader
    parse_commandline()
    train_loader, val_loader, _ = build_data()  # Generate the train and validate loaders

    # create raw model and loss func
    model, loss_func = build_model()

    # Start training
    model_name = petname.generate()  # Give a unique name to the model
    print(f'Start training for model {model_name}...')
    if Config.both_series:
        save_dir = f'./Trained_Models/both/{Config.model_num}/{model_name}'  # Directory to save the model
    else:
        save_dir = f'./Trained_Models/{Config.one_series}/{Config.model_num}/{model_name}'  # Directory to save the model
    os.makedirs(save_dir)
    save_run_params_in_file(save_dir, Config)  # Save the relevant configurations for the model

    # Build the training class
    model_trainer = TrainUtils(model=model,
                               model_num=Config.model_num,
                               loss_func=loss_func,
                               epochs=Config.epochs,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               save_dir=save_dir,
                               )
    # Start the training
    print(f"Starting fresh training...with model {model_name}")
    model_trainer.start_training()


@contextmanager
def cuda_context(cuda=None):
    """
    A context manager to use cuda (if available) as the default tensor type.
    New tensors will be created on the specified device automatically.
    """
    if cuda is None:
        cuda = torch.cuda.is_available()
    old_tensor_type = torch.cuda.FloatTensor if torch.tensor(0).is_cuda else torch.FloatTensor
    old_generator = torch.default_generator
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    torch.default_generator = torch.Generator('cuda' if cuda else 'cpu')
    yield
    torch.set_default_tensor_type(old_tensor_type)
    torch.default_generator = old_generator


if __name__ == "__main__":
    with cuda_context():
        main()
