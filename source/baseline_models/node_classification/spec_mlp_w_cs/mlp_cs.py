#alg_name = 'mlp_cs'

import os

os.environ["OMP_NUM_THREADS"] = "2"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "2"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "2"  # export NUMEXPR_NUM_THREADS=1

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d

import sys

sys.path.append('../../../')

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import models

import torch_geometric.transforms as T
from torch_geometric.nn.models import CorrectAndSmooth

from logger import Logger
import argparse

from torch.utils.tensorboard import SummaryWriter

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import pandas as pd
import seaborn as sn
import time


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.lins = ModuleList([Linear(in_channels, hidden_channels)])
        self.bns = ModuleList([BatchNorm1d(hidden_channels)])

        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        self.lins.append(Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lins in self.lins:
            lins.reset_parameters()
        for bns in self.bns:
            bns.reset_parameters()

    def forward(self, x):
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = bn(lin(x).relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


def spectral(data, post_fix):
    '''
    generate spectral embeddings, save the results in ./embeddings/spectral{post_fix}.pt
    '''
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./norm_spec.jl")

    print('Setting up spectral embedding', flush=True)
    adj = data.adj_t
    adj = adj.to_scipy(layout='csr')
    result = torch.tensor(Main.main(adj, 128)).float()
    print('Done!', flush=True)

    torch.save(result, f'./embeddings/spectral{post_fix}.pt')

    return result


def process_adj(data, device):
    adj_t = data.adj_t.to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

    return DAD, DA


def train(model, optimizer, x_train, criterion, y_train):
    model.train()
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train.view(-1))
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=None):
    model.eval()
    out = model(x) if out is None else out
    pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y[train_idx],
        'y_pred': pred[train_idx]
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y[val_idx],
        'y_pred': pred[val_idx]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[test_idx],
        'y_pred': pred[test_idx]
    })['acc']
    return train_acc, val_acc, test_acc, out


@torch.no_grad()
def extended_test(model, x, y_true, split_idx, metric_dict, unique_labels, out=None):
    model.eval()
    out = model(x) if out is None else out
    y_pred = out.argmax(dim=-1, keepdim=True)

    # initialize dictionary for metric results
    metric_res_dict = {'train': {}, 'valid': {}, 'test': {}}
    for metric_id, method_info in metric_dict.items():
        scorer_fct = method_info['function']
        scorer_kwargs = method_info['kwargs']
        try:
            metric_res_dict['train'][metric_id] = scorer_fct(y_true[split_idx['train']].cpu(),
                                                             y_pred[split_idx['train']].cpu(),
                                                             **scorer_kwargs)
        except:
            metric_res_dict['train'][metric_id] = scorer_fct(
                skp.label_binarize(y_true[split_idx['train']].cpu(), classes=unique_labels),
                skp.label_binarize(y_pred[split_idx['train']].cpu(), classes=unique_labels),
                **scorer_kwargs)

        try:
            metric_res_dict['valid'][metric_id] = scorer_fct(y_true[split_idx['valid']].cpu(),
                                                             y_pred[split_idx['valid']].cpu(),
                                                             **scorer_kwargs)
        except:
            metric_res_dict['valid'][metric_id] = scorer_fct(
                skp.label_binarize(y_true[split_idx['valid']].cpu(), classes=unique_labels),
                skp.label_binarize(y_pred[split_idx['valid']].cpu(), classes=unique_labels),
                **scorer_kwargs)

        try:
            metric_res_dict['test'][metric_id] = scorer_fct(y_true[split_idx['test']].cpu(),
                                                            y_pred[split_idx['test']].cpu(),
                                                            **scorer_kwargs)
        except:
            metric_res_dict['test'][metric_id] = scorer_fct(
                skp.label_binarize(y_true[split_idx['test']].cpu(), classes=unique_labels),
                skp.label_binarize(y_pred[split_idx['test']].cpu(), classes=unique_labels),
                **scorer_kwargs)

    figure_size = (7.315, 4.13)
    # generate confusion matrix
    cf_matrix = confusion_matrix(y_true[split_idx['valid']].cpu(), y_pred[split_idx['valid']].cpu())
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100)
    cf_fig = plt.figure(figsize=figure_size, dpi=150)
    sn.heatmap(df_cm, annot=True)
    metric_res_dict['valid']['cf_matrix'] = cf_fig

    # generate histogram
    bins = y_true.unique()
    bins = torch.cat((bins, bins[-1].unsqueeze(-1) + 1), 0)
    bins = bins.cpu()
    hist_pred_fig, ax = plt.subplots(figsize=figure_size, dpi=150)
    ax.hist(np.array(y_pred[split_idx['valid']].cpu()), log=True, bins=bins)
    metric_res_dict['valid']['hist_pred'] = hist_pred_fig

    # hist_true_fig, ax = plt.subplots(figsize=figure_size, dpi=150)
    # ax.hist(np.array(y_true[split_idx['valid']].cpu()), log=True, bins=bins)
    # metric_res_dict['valid']['hist_true'] = hist_true_fig

    return metric_res_dict, out


def calculate_weight_vector(labels, return_unique_labels=False):
    n_labels = labels.shape[0]
    unique_labels, counts = labels.unique(return_counts=True)
    weights = n_labels / counts  # 1 - (counts / n_labels)
    if return_unique_labels:
        return weights, unique_labels.cpu()
    else:
        return weights


def main():
    # get start time
    start_time = time.clock()

    parser = argparse.ArgumentParser(description='OGBN-Products (MLP-CS)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--use_embed", action="store_true")
    parser.add_argument("--use_cached", action="store_true")
    parser.add_argument('--log_dir', type=str, default='../tensorboard_logs')
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--n_par_combs', type=int, default=0)
    parser.add_argument('--curr_param_idx', type=int, default=0)
    parser.add_argument('--n_stop_train', type=int, default=160)
    parser.add_argument('--ex', type=int, default=-1)
    parser.add_argument('--model_states', type=str)
    args = parser.parse_args()
    print(args, flush=True)

    if args.use_embed:
        alg_name = 'mlp_cs_node2vec'
    else:
        alg_name = 'mlp_cs'

    # dictionary with desired metrics:
    #       keys:   own metric id/name;
    #       values: dictionary with function pointer and keyword arguments the should be passed to the given function
    #       the first item in dict is used for logging information displayed in command line
    metric_dict = {'balanced_accuracy': {'function': skm.balanced_accuracy_score,
                                         'kwargs': {}},
                   'f1_micro': {'function': skm.f1_score,
                                'kwargs': {'average': 'micro'}},
                   'f1_weighted': {'function': skm.f1_score,
                                   'kwargs': {'average': 'weighted'}},
                   'accuracy': {'function': skm.accuracy_score,
                                'kwargs': {}},
                   'roc_auc_ovr': {'function': skm.roc_auc_score,
                                   'kwargs': {'multi_class': 'ovr'}},
                   'roc_auc_ovo': {'function': skm.roc_auc_score,
                                   'kwargs': {'multi_class': 'ovo'}},
                   'roc_auc_ovr_weighted': {'function': skm.roc_auc_score,
                                            'kwargs': {'multi_class': 'ovr', 'average': 'weighted'}},
                   'roc_auc_ovo_weighted': {'function': skm.roc_auc_score,
                                            'kwargs': {'multi_class': 'ovo', 'average': 'weighted'}}}

    dataset = PygNodePropPredDataset(name=f'ogbn-{args.ds_name}',
                                     root='./dataset/',
                                     transform=T.ToSparseTensor())
    print(dataset, flush=True)
    # evaluator = Evaluator(name=f'ogbn-{args.ds_name}')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    print(data, flush=True)

    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else 'cpu')

    # generate and add embeddings
    if args.use_embed:
        if args.use_cached:
            embeddings = torch.load(f'embeddings/{args.ds_name}_embedding.pt', map_location='cpu')
        else:
            embeddings = spectral(data, 'products')
        data.x = torch.cat([data.x, embeddings], dim=-1)

    x, y = data.x.to(device), data.y.to(device)

    # MLP-Wide
    model = MLP(x.size(-1),
                dataset.num_classes,
                hidden_channels=args.hidden_channels,
                num_layers=args.num_layers,
                dropout=args.dropout).to(device)

    # calculate weights for loss function and fetch unique labels
    weights, unique_labels = calculate_weight_vector(data.y, return_unique_labels=True)
    # put weights tensor to defined device
    weights = weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    train_idx = split_idx['train'].to(device)
    val_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)
    x_train, y_train = x[train_idx], y[train_idx]

    logger = Logger(args.runs, args)

    ####################################################################################################################
    #                               Directory Generateion Part1
    ####################################################################################################################
    # determine algorithm logged algorithm executions and define execution directory
    ex_prefix = 'ex'
    # check if logging root directory exists
    # define log dir path
    log_dir = os.path.join(args.log_dir, args.ds_name, alg_name)
    if not os.path.exists(log_dir):
        # if not generate it
        os.makedirs(log_dir)

    if args.ex == -1:
        # fetch list of directories in logging root directory
        dir_entries = os.listdir(log_dir)
        # generate list of sub directories with the defined subdirectory prefix
        prev_logs = [name for name in dir_entries if name[:2] == ex_prefix]
        # check if there exist subdirectories with the defined prefix
        if len(prev_logs) > 0:
            # if so: set the index to the next higher number
            prev_logs.sort(reverse=True)
            curr_dir_idx = int(prev_logs[0][-2:]) + 1
        else:
            # if not: set index to 0
            curr_dir_idx = 0
    else:
        curr_dir_idx = args.ex

    # stitch resulting directory for tensorboard logging data
    base_log_dir = os.path.join(log_dir, f'{ex_prefix}{curr_dir_idx:0>2}')

    # define logging sub_directory name suffix for summary writer (index of current parameter_combination
    if args.n_par_combs > 1:
        base_log_dir = os.path.join(base_log_dir, f'{args.curr_param_idx:0>3}of{args.n_par_combs}')
    ####################################################################################################################
    ####################################################################################################################

    # train loop
    for run in range(args.runs):

        # add current run to parameter class
        args.cur_run = run

        ################################################################################################################
        #                               Directory Generateion Part2
        ################################################################################################################
        if args.runs > 1:
            log_dir = os.path.join(base_log_dir, f'run{run}')
        else:
            log_dir = base_log_dir

        # make directory for best model state dict
        model_state_dict_dir = os.path.join(log_dir, 'model_state_dicts')
        # check if logging root directory exists
        if not os.path.exists(model_state_dict_dir):
            # if not generate it
            os.makedirs(model_state_dict_dir)
        ################################################################################################################
        ################################################################################################################

        # instantiate tensorboard writer
        writer = SummaryWriter(log_dir)

        if args.model_states is not None:
            state_dict = torch.load(args.model_states)
            model.load_state_dict(state_dict)
        else:
            model.reset_parameters()

        # initialize dictionary of metric hightscores with 0
        best_metric_dict = {key: 0.0 for key in metric_dict.keys()}
        # initialize counter for epochs without updating any metric highscore
        n_eval_no_best = 0

        print(sum(p.numel() for p in model.parameters()), flush=True)

        print('', flush=True)
        print(f'Run {run + 1:02d}:', flush=True)
        print('', flush=True)

        best_val_acc = 0
        for epoch in range(1, args.epochs + 1):  ##
            loss = train(model, optimizer, x_train, criterion, y_train)

            # tensorboard: log train loss
            writer.add_scalar("Loss/train", loss, epoch)
            # train_acc, val_acc, test_acc, out = test(model, x, evaluator, y, train_idx, val_idx, test_idx)

            if epoch > 9 and epoch % args.eval_steps == 0:

                metric_res_dict, out = extended_test(model, x, y,
                                                     {'train': train_idx, 'valid': val_idx, 'test': test_idx},
                                                     metric_dict, unique_labels)

                result = [list(metric_res_dict['train'].values())[0], list(metric_res_dict['valid'].values())[0],
                          list(metric_res_dict['test'].values())[0]]

                train_acc, val_acc, test_acc = result

                # iterate over passed metrics and add their performance values to tensorboard logging data
                for metric in metric_dict.keys():
                    writer.add_scalars(f"{metric}",
                                       {'train': metric_res_dict['train'][metric],
                                        'valid': metric_res_dict['valid'][metric],
                                        'test': metric_res_dict['test'][metric]}, epoch)

                writer.add_figure("Histogram/Logits", metric_res_dict['valid']['hist_pred'], epoch)
                # writer.add_figure("Histogram/Labels", metric_res_dict['valid']['hist_true'], epoch)
                # generate confusion matrix
                writer.add_figure("Confusion Matrix", metric_res_dict['valid']['cf_matrix'], epoch)

                # check if any metric achieves a new highscore
                flag_updated_highscore = 0
                for key, value in best_metric_dict.items():
                    if metric_res_dict['valid'][key] > value:
                        print(f'update best valid {key} to: ', metric_res_dict['valid'][key])
                        best_metric_dict[key] = float(metric_res_dict['valid'][key])
                        # save model weights
                        torch.save(model.state_dict(), os.path.join(model_state_dict_dir,
                                                                    f'best_{key}_model_state_dict.pth'))
                        flag_updated_highscore = 1

                # check if an metric achieved a new peak performance
                if flag_updated_highscore == 1:
                    # if so: set counter of epochs without performance improvement to 0
                    n_eval_no_best = 0
                else:
                    # increase conter of epochs without performance improvemetn by one
                    n_eval_no_best += 1
                    print(f"Epochs without updating best valid {key} score: {n_eval_no_best * args.eval_steps}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    y_soft = out.softmax(dim=-1)

                print(
                    f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                    f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
                    flush=True)
                # writer.add_scalar("Accuracy/train", train_acc, epoch)
                # writer.add_scalar("Accuracy/valid", val_acc, epoch)
                # writer.add_scalar("Accuracy/test", test_acc, epoch)

                if epoch > 70 and n_eval_no_best * args.eval_steps > args.n_stop_train:
                    break

        DAD, DA = process_adj(data, device)

        post = CorrectAndSmooth(num_correction_layers=50,
                                correction_alpha=1.0,
                                num_smoothing_layers=50,
                                smoothing_alpha=0.8,
                                autoscale=False,
                                scale=15.)

        print('Correct and smooth...', flush=True)
        y_soft = post.correct(y_soft, y_train, train_idx, DAD)
        y_soft = post.smooth(y_soft, y_train, train_idx, DA)
        print('Done!', flush=True)

        #cs_metric_res_dict = {'cs_train': {}, 'cs_valid': {}, 'cs_test': {}}

        cs_metric_res_dict, _ = extended_test(model, x, y, {'train': train_idx, 'valid': val_idx, 'test': test_idx},
                                        metric_dict, unique_labels, out=y_soft)

        cs_metric_res_dict = {f'cs_{key}': value for key, value in cs_metric_res_dict.items()}

        result = [list(cs_metric_res_dict['cs_train'].values())[0], list(cs_metric_res_dict['cs_valid'].values())[0],
                  list(cs_metric_res_dict['cs_test'].values())[0]]

        train_acc, val_acc, test_acc = result

        print(
            f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
            flush=True)

        logger.add_result(run, result)
        best_metric_dict.update({'loss': loss, 'trained_epochs': epoch,
                                 'diff_train_valid_acc': float(train_acc - val_acc),
                                 'execution_time': time.clock() - start_time})
        hparm_cs_valid_res = {}
        for key in metric_dict.keys():
            hparm_cs_valid_res[f'cs_{key}'] = cs_metric_res_dict['cs_valid'][key]

        best_metric_dict.update(hparm_cs_valid_res)
        hparams_metric_dict = {f'hparam/{key}': value for key, value in best_metric_dict.items()}
        writer.add_hparams(
            vars(args),
            hparams_metric_dict)
        writer.flush()
        writer.close()

    # logger.print_statistics()


if __name__ == '__main__':
    main()
