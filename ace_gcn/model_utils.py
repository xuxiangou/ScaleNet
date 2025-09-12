#Import necessary modules
"""
@author: pgg
"""
import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import pickle 
from tqdm import tqdm

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch_scatter import scatter_mean, scatter_add
from generate_dataset import GraphData, collate_pool
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import pprint 

from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
torch.manual_seed(42);

#----- PLOTTING PARAMS ----# 
import seaborn as sns; 
sns.color_palette('colorblind')

import matplotlib
matplotlib.use('pdf') #To avoid tkinter.TclError on HPC: couldn't connect to display error

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

plot_params = {
'font.size' : 10,
'axes.titlesize' : 14,
'axes.labelsize' : 10,
'axes.labelweight' : 'bold',
#'lines.linewidth' : 3,
#'lines.markersize' : 10,
'xtick.labelsize' : 10,
'ytick.labelsize' : 10,
"legend.loc":'best' 
} 
plt.rcParams.update(plot_params)

    
def get_dataloader(args, dataset, indexlist):
    """
    Generate a dataloader class for training the PyTorch nn
    User defined collate pool used 
    Change num_workers to get advantage of the multi-cores 
    """

    _sampler = SubsetRandomSampler(indexlist)
    _loader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=_sampler,
                                num_workers=1,
                                collate_fn=collate_pool)
    return _loader 

def save_checkpoint(state, is_best, directory_path, text_entry):
    temp_file_path = '{0}/checkpoint_train_{1}.pth.tar'.format(directory_path, text_entry)
    best_file_path = '{0}/best_model_train_{1}.pth.tar'.format(directory_path, text_entry)
    torch.save(state, temp_file_path)
    if is_best:
        shutil.copyfile(temp_file_path, best_file_path)

def load_cgcnn_model(model_path, model_instance=False):
    """Load model and related params"""
    print("=> loading model '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    normalizer_test = Normalizer(torch.zeros(3))
    normalizer_test.load_state_dict(checkpoint['normalizer'])
    args = argparse.Namespace(**checkpoint['args'])

    print("=> loaded model '{}' (epoch {}, validation {})"
        .format(model_path, checkpoint['epoch'], checkpoint['best_mae_error']))

    if model_instance:
        model_instance.load_state_dict(checkpoint['state_dict'])
        return model_instance, normalizer_test, args 
    else:
        return checkpoint, normalizer_test

class Normalizer(object):
    """
    Normalize a Tensor and restore it later. This is used during training the model.
    """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def generate_dataset_list(_root_dir, _id_prop_name, _pickle_path, _pickle=True):
    '''
    Convenience function to generate a list of Graph Object
    This makes it easy for splitting later for train/val/test 
    '''

    print('---- Loading data ----')
    data_graph_object = GraphData(root_dir = _root_dir, 
                    id_prop_filename = _id_prop_name,
                    pickle_path = _pickle_path,
                    save_pickle = _pickle)

    #print(data_graph_object)
    print(len(data_graph_object))
    _idx = np.arange(len(data_graph_object))
    data_list = []

    '''
    This loop is bottle neck as __getitem__ might take time if pickle not saved
    '''
    for idx in tqdm(_idx):
        try:
            data_list.append(data_graph_object[idx])
        except (ValueError, IndexError):
            print('------ skipping due to error -------')
            continue 
    return data_list

def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_, target, _) in enumerate(train_loader):
        # measure data loading time -- input is again the CIFdata OUPUT 
        data_time.update(time.time() - end)
        if args.cuda == True:
            input_var = (tensor.to('cuda') for tensor in input_)
            
        else:
            input_var = input_

        # normalize target
        target_normed = normalizer.norm(target)
        if args.cuda == True:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error.item(), target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad() #Refresh the grad array 
        loss.backward() #Compute the backward prop 
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and epoch % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                   epoch+1, i+1, len(train_loader),loss=losses, mae_errors=mae_errors)
                 )

    return mae_errors, losses

def validate(args, val_loader, model, criterion, epoch, normalizer, _model_save_path, output_file_name='test_cases', test=False, make_csv=False):
    #Only forward pass here Duh! 
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda == True:
            with torch.no_grad():
                input_var = (tensor.to("cuda") for tensor in input_)
        else:
            with torch.no_grad():
                input_var = input_

        target_normed = normalizer.norm(target)
        
        if args.cuda == True:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        with torch.no_grad():
          output = model(*input_var)
          loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error.item(), target.size(0))
        
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0  and epoch % args.print_freq == 0:
            print('Val => [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                   i+1, len(val_loader),loss=losses, mae_errors=mae_errors)
                 )

    if test and make_csv:
        star_label = '**'
        import csv
        with open('{0}/{1}.csv'.format(_model_save_path, output_file_name), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
                
    return mae_errors, losses, mae_errors.avg

def parity_plot(csv_file,plotname=None, saveplot=False, place_holder=0.2, plot=False):
    data_df_plotting = pd.read_csv('{0}.csv'.format(csv_file), sep=',',header=None,names=['outcar_id','actual_value','predicted_value'])
    total_ads = []
    for index, row in data_df_plotting.iterrows():
        num_OH_ending = row['outcar_id'].split('_')[-1]
        total_ads.append(num_OH_ending)
    
    data_df_plotting['ads_OH'] = total_ads

    # Plot  
    x_lims=[min(data_df_plotting['actual_value'])-0.1,max(data_df_plotting['actual_value'])+0.1]
    y_lims=[min(data_df_plotting['predicted_value'])-0.1,max(data_df_plotting['predicted_value'])+0.1]

    # Calculate the error metrics
    mae = mean_absolute_error(data_df_plotting['actual_value'], data_df_plotting['predicted_value'])
    rmse = np.sqrt(mean_squared_error(data_df_plotting['actual_value'], data_df_plotting['predicted_value']))
    r2 = r2_score(data_df_plotting['actual_value'], data_df_plotting['predicted_value'])
    rel_error = (data_df_plotting['predicted_value'] - data_df_plotting['actual_value']) / data_df_plotting['actual_value'] * 100
    mape = rel_error.abs().mean()
    text = ('  n = {0}\n'
            '  MAE = {1:0.2f} eV\n'
            '  RMSE = {2:0.2f} eV\n'
            '  R$^2$ = {3:0.2f}\n'
            '  MAPE = {4:0.2f}%'.format(len(data_df_plotting), mae, rmse, r2, mape))

    if plot: 
        g = sns.jointplot(data=data_df_plotting, x="actual_value", y="predicted_value", hue="ads_OH")
        ax = g.ax_joint
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.plot(x_lims, x_lims, 'k--')
        ax.set_xlabel('DFT Avg $BE_{OH}$ [eV]')
        ax.set_ylabel('CGCNN Avg $BE_{OH}$ [eV]')
        ax.legend
        ax.text(x=x_lims[1]-place_holder, y=y_lims[0]+place_holder, s=text, fontsize=12,
                    horizontalalignment='left',
                    verticalalignment='top')
    if saveplot: 
        plt.savefig('{}.png'.format(plotname),dpi=300)
    
    return mape, mae, rmse


def save_convergence(path, epochs, train_mae_errors, train_losses, val_mae_errors, val_losses, filename):
    """save mae, loss for each epoch to csv"""
    val = lambda x_list: [x.val for x in x_list]
    train_mae, val_mae = val(train_mae_errors), val(val_mae_errors)
    train_loss, val_loss = val(train_losses), val(val_losses)
    df = pd.DataFrame({'epoch': epochs,
                       'train_mae': train_mae, 'val_mae': val_mae,
                       'train_loss': train_loss, 'val_loss': val_loss})
    df.to_csv('{}/convergence_mae_loss_{}.csv'.format(path,filename))

def set_axis_style(ax, xlabel, ylabel, title):
    """
    set axis label sizes
    """
    ax.tick_params(labelsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=10)


def convergence_plot(path, filename, rolling=False):
    try:
        data = pd.read_csv('{}/convergence_mae_loss_{}.csv'.format(path, filename), sep=',', delimiter=None, usecols=range(1,6))
    except OSError:
        if not os.path.exists('{}/convergence_mae_loss.csv'.format(path)):
            raise

    fig = plt.figure()
    #fig.subplots_adjust(left=0.1, right=1.5, top=2.0, bottom=0.1, hspace=0.5)
    ax1 = fig.add_subplot(211)

    if rolling:
        data_aug = data.rolling(11, center=True).mean()
    else:
        data_aug = data

    ax1.plot(data_aug['epoch'], data_aug['train_mae'], label='train')
    ax1.plot(data_aug['epoch'], data_aug['val_mae'], label='valid')
    set_axis_style(ax1, 'Epochs', 'MAE', 'Epochs vs MAE')

    ax2 = fig.add_subplot(212)
    ax2.plot(data_aug['epoch'], data_aug['train_loss'], label='train')
    ax2.plot(data_aug['epoch'], data_aug['val_loss'], label='valid')
    set_axis_style(ax2, 'Epochs', 'Loss', 'Epochs vs RMSE')
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(path, filename), bbox_inches='tight', dpi=300)

def make_parity(args, _loader, _model_save_path, _model, _criterion, _epoch, _normalizer, _csv_file, _plot_name, _space, _saveplot=True):
    _, _, _ = validate(args, _loader, _model, _criterion, _epoch, _normalizer, _model_save_path, output_file_name=_csv_file, make_csv=True, test=True)
    csv_file = os.path.join(_model_save_path, _csv_file)
    plotname = os.path.join(_model_save_path, _plot_name)
    mape, mae, rmse = parity_plot(csv_file, plotname, saveplot=_saveplot, place_holder=_space, plot=True)
    return mape, mae, rmse
