import sys
import matplotlib.pyplot as plt
import torch

from utils import getxy


class TrainUtils:
    def __init__(self,
                 model,
                 model_num,
                 loss_func,
                 epochs,
                 train_loader,
                 val_loader,
                 save_dir,
                 ):
        """
        This class handles the training of a pytorch model.
        :param model (nn.Module): model to be trained
        :param model_num (int): number specifying the model being trained
        :param loss_func (function): loss function that has the following parameters (input x, target y, model, pred).
        Target is a must. Either model or pred has to be given. If model is given, input must be given.
        :param epochs (int): epochs to train
        :param train_loader (utils.data.DataLoader): training data
        :param val_loader (utils.data.DataLoader): validation data
        :param save_dir (path):  path to save the best model parameters
        """
        self.model = model
        self.model_num = model_num
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(model.parameters())
        self.loss = loss_func
        self.epochs = epochs
        self.best_val_loss = sys.float_info.max
        self.save_dir = save_dir

        self.loss_list_train = []  # List to collect the loss per epoch
        self.loss_list_val = []  # List to collect the loss per epoch

    def start_training(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_step()
            self.loss_list_train.append(train_loss)
            print(f'Train loss for epoch {epoch} is {train_loss}')

            val_loss = self.val_step()
            self.loss_list_val.append(val_loss)
            print(f'Validation loss for epoch {epoch} is {val_loss}')

            # Saving the loss function
            plt.plot(self.loss_list_train)
            plt.plot(self.loss_list_val)
            i = int(len(self.loss_list_train) * 0.1)
            plt.ylim(
                min(*self.loss_list_train[i:], *self.loss_list_val[i:]),
                max(*self.loss_list_train[i:], *self.loss_list_val[i:]),
            )
            plt.legend(['Train loss', 'Validate loss'])
            plt.savefig(f"{self.save_dir}/loss.png")
            plt.close()

            # Saving the model if the current val loss is lower than the best val loss so far
            # Prevents saving an overfitted model
            if val_loss < self.best_val_loss:
                print(f'Saving best model so far, epoch {epoch}')
                self.best_val_loss = val_loss
                self.save_model()

        # Saving the losses
        with open(f'{self.save_dir}/loss.csv', mode='w') as f:
            f.writelines(
                f"{train_loss}, {val_loss}\n"
                for train_loss, val_loss in zip(self.loss_list_train, self.loss_list_val)
            )

    def train_step(self):
        self.model.train()  # train mode
        running_batch_loss = 0
        for step, batch in enumerate(self.train_loader, 1):
            x, y = getxy(batch)
            try:
                loss = self.loss(x, y, self.model)
                self.optimizer.zero_grad()  # reset gradients
                loss.backward()  # backpropagation

                # gradient clipping
                _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()  # apply optimization step
                running_batch_loss += loss.item()
            except Exception as e:
                print(e, file=sys.stderr)
                continue

        return running_batch_loss

    def val_step(self):
        self.model.eval()  # validate mode
        running_batch_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader, 1):
                x, y = getxy(batch)
                try:
                    loss = self.loss(x, y, self.model)
                    running_batch_loss += loss.item()
                except Exception as e:
                    print(e, file=sys.stderr)
                    continue
        return running_batch_loss

    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_dir}/model.pt')
