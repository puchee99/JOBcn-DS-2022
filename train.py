import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from model import  MulticlassSimpleClassification, init_weights
from dataset import create_datasets, get_dataframe, get_path, get_train_test_val, get_train_test_val_variable
from utils import SaveBestModel, save_model, save_plots, save_plot_cm, save_plot_roc, plot_features,str2bool, logging_loader
save_best_model = SaveBestModel()
logger = logging_loader()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Computation device: {device}\n")

#https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=4000,
    help='number of epochs to train our network for')
parser.add_argument('-lr', '--learningrate', type=float, default=0.0007,#1e-4,
    help='lerning rate number to train our network')
parser.add_argument('-bl', '--batchlearning', type=str2bool, default=False,
    help='if true creates DataLoaders to use batch in training')
parser.add_argument('-trf', '--trainfile', type=str, default='train.csv',
    help='Name of the training file')
parser.add_argument('-l', '--label', type=str, default='Insect',
    help='Name of the label column')
parser.add_argument('-tf', '--tensorboard', type=str2bool, default=False,
    help='save logs to use in tensorboard')
parser.add_argument('-mp', '--makeplots', type=str2bool, default=True,
    help='make some plots and save to output/plots folder')
parser.add_argument('-r', '--run_test', type=str2bool, default=True,
    help='run train test val accuracy')
parser.add_argument('-rr', '--test_without_label', type=str2bool, default=True,
    help='predict target and save results without knowing y_pred true label')

args = vars(parser.parse_args())

LABEL_NAME = args['label']
TRAIN_FILE = args['trainfile']
LR = args['learningrate']
EPOCHS = args['epochs']

if args['tensorboard']:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('output/log_runs/insects_experiment_1')
if args['makeplots']:
    path, extension = get_path('train.csv')
    df = get_dataframe(path, extension)
    plot_features(df,df['Insect'])

def train(model, X, y, optimizer, criterion):
    model.train()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    correct_train = (torch.argmax(y_pred, dim=1) == y).type(torch.FloatTensor)
    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), correct_train.mean()

def validate(model, X_test, y_test, criterion):
    model.eval()
    y_pred_val = model(X_test)
    val_loss = criterion(y_pred_val, y_test)
    correct = (torch.argmax(y_pred_val, dim=1) == y_test).type(torch.FloatTensor)
    return val_loss.item(), correct.mean()

def run_model(model,X_train,X_test,y_train, y_test):
    print_model_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    criterion   = torch.nn.CrossEntropyLoss()
    train_loss     = np.zeros((EPOCHS,))
    train_acc = np.zeros((EPOCHS,))
    valid_loss     = np.zeros((EPOCHS,))
    valid_acc = np.zeros((EPOCHS,))
    last_loss, dif_last_loss, epoch = np.inf, 1e5, 0

    while dif_last_loss > 0.000000001 and epoch < EPOCHS:
        train_epoch_loss, train_epoch_acc = train(model, X_train,y_train, optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, X_test, y_test, criterion)
        train_loss[epoch], valid_loss[epoch] = train_epoch_loss, valid_epoch_loss
        train_acc[epoch], valid_acc[epoch] = train_epoch_acc, valid_epoch_acc
        dif_last_loss = np.abs(last_loss - valid_epoch_loss)
        last_loss = valid_epoch_loss
        
        # save the best model till now if we have the least loss in the current epoch
        save_best_model(
            valid_epoch_loss, epoch, model, optimizer, criterion
        )
        if (epoch+1) % 200 == 0:
            print("Epoch [{}/{}],  Loss: {:.4f}  Accuracy: {:.4f} ".format(epoch+1, EPOCHS,  valid_epoch_loss, valid_epoch_acc))
            logger.info("Epoch [{}/{}],  Loss: {:.4f}  Accuracy: {:.4f} ".format(epoch+1, EPOCHS,  valid_epoch_loss, valid_epoch_acc))
        epoch+=1

    save_model(epoch, model, optimizer, criterion)
    if args['tensorboard']:
        writer.add_scalar('training loss',last_loss,epoch)
    logger.info("Finish training model \n")
    save_plots(train_acc[:epoch], valid_acc[:epoch], train_loss[:epoch], valid_loss[:epoch],str(type(model)).split("'")[1].split(".")[-1])# model.name)
    save_plot_cm(X_test, y_test, model)
    save_plot_roc(X_test, y_test, model)
    return valid_acc[:epoch], valid_loss[:epoch], model 


#--------------------------aux--------------------------------------
def print_model_params(model):
    logger.info('Running Model\n: {}'.format(model))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"{total_trainable_params:,} training parameters.")
    return
    
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return torch.round(acc * 100)

if __name__ == "__main__":
    df_train, y, col_names = create_datasets(TRAIN_FILE, LABEL_NAME)
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val(df_train,y)
    X_train_t, X_test_t, X_val_t, y_train_t, y_test_t, y_val_t = get_train_test_val_variable(X_train, X_test, X_val, y_train, y_test, y_val)
    model     = MulticlassSimpleClassification(X_train.shape[1], 3).to(device)
    model.apply(init_weights)
    run_model(model, X_train_t, X_test_t, y_train_t, y_test_t)
    if args['run_test'] or args['test_without_label']:
        os.system(f"python test.py -v {args['run_test']} -r {args['test_without_label']}")
    
