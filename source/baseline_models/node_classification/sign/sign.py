alg_name = 'sign'

import os

os.environ["OMP_NUM_THREADS"] = "2"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "2"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "2"  # export NUMEXPR_NUM_THREADS=1

import sys

sys.path.append('../../../')

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.transforms import SIGN

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.lin = torch.nn.Linear((num_layers + 1) * hidden_channels,
                                   out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, xs):
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=0.5, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.lin(x)
        return torch.log_softmax(x, dim=-1)


def train(model, xs, y_true, optimizer, device, weights):
    model.train()

    optimizer.zero_grad()
    out = model(xs)
    loss = F.nll_loss(out, y_true.squeeze(1), weight=weights)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, xs, y_true, evaluator):
    model.eval()

    y_preds = []
    loader = DataLoader(range(y_true.size(0)), batch_size=400000)
    for perm in loader:
        y_pred = model([x[perm] for x in xs]).argmax(dim=-1, keepdim=True)
        y_preds.append(y_pred.cpu())
    y_pred = torch.cat(y_preds, dim=0)

    return evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })['acc']

@torch.no_grad()
def extended_test(model, data_dict, label_dict, device, metric_dict, unique_labels):
    model.eval()
    y_pred = {'train': [], 'valid': [], 'test': []}
    for key, y_true in label_dict.items():
        xs = data_dict[key]
        y_preds = []
        loader = DataLoader(range(y_true.size(0)), batch_size=400000)
        for perm in loader:
            y_pred[key] = model([x[perm] for x in xs]).argmax(dim=-1, keepdim=True)
            y_preds.append(y_pred[key].cpu())
        y_pred[key] = torch.cat(y_preds, dim=0)

    # initialize dictionary for metric results
    metric_res_dict = {'train': {}, 'valid': {}, 'test': {}}
    for metric_id, method_info in metric_dict.items():
        scorer_fct = method_info['function']
        scorer_kwargs = method_info['kwargs']
        try:
            metric_res_dict['train'][metric_id] = scorer_fct(label_dict['train'].cpu(), y_pred['train'],
                                                             **scorer_kwargs)
        except:
            metric_res_dict['train'][metric_id] = scorer_fct(
                skp.label_binarize(label_dict['train'].cpu(), classes=unique_labels),
                skp.label_binarize(y_pred['train'], classes=unique_labels),
                **scorer_kwargs)

        try:
            metric_res_dict['valid'][metric_id] = scorer_fct(label_dict['valid'].cpu(), y_pred['valid'],
                                                             **scorer_kwargs)
        except:
            metric_res_dict['valid'][metric_id] = scorer_fct(
                skp.label_binarize(label_dict['valid'].cpu(), classes=unique_labels),
                skp.label_binarize(y_pred['valid'], classes=unique_labels),
                **scorer_kwargs)

        try:
            metric_res_dict['test'][metric_id] = scorer_fct(label_dict['test'].cpu(), y_pred['test'],
                                                            **scorer_kwargs)
        except:
            metric_res_dict['test'][metric_id] = scorer_fct(
                skp.label_binarize(label_dict['test'].cpu(), classes=unique_labels),
                skp.label_binarize(y_pred['test'], classes=unique_labels),
                **scorer_kwargs)

    figure_size = (7.315, 4.13)
    # generate confusion matrix
    cf_matrix = confusion_matrix(label_dict['valid'].cpu(), y_pred['valid'])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100)
    cf_fig = plt.figure(figsize=figure_size, dpi=150)
    sn.heatmap(df_cm, annot=True)
    metric_res_dict['valid']['cf_matrix'] = cf_fig

    # generate histogram
    bins = label_dict['train'].cpu().unique()
    bins = torch.cat((bins, bins[-1].unsqueeze(-1) + 1), 0)
    hist_pred_fig, ax = plt.subplots(figsize=figure_size, dpi=150)
    ax.hist(np.array(y_pred['valid']), log=True, bins=bins)
    metric_res_dict['valid']['hist_pred'] = hist_pred_fig

    # hist_true_fig, ax = plt.subplots(figsize=figure_size, dpi=150)
    # ax.hist(np.array(label_dict['valid'].cpu()), log=True, bins=bins)
    # metric_res_dict['valid']['hist_true'] = hist_true_fig

    return metric_res_dict


def calculate_weight_vector(labels, return_unique_labels=False):
    n_labels = labels.shape[0]
    unique_labels, counts = labels.unique(return_counts=True)
    weights = n_labels / counts  # 1 - (counts / n_labels)
    if return_unique_labels:
        return weights, unique_labels
    else:
        return weights


def main():
    # get start time
    start_time = time.time()

    parser = argparse.ArgumentParser(description='ogbn- SIGN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='../tensorboard_logs')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_par_combs', type=int, default=0)
    parser.add_argument('--curr_param_idx', type=int, default=0)
    parser.add_argument('--n_stop_train', type=int, default=160)
    parser.add_argument('--ex', type=int, default=-1)
    parser.add_argument('--model_states', type=str)
    parser.add_argument('--test_only',action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

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
                                            'kwargs': {'multi_class': 'ovo', 'average': 'weighted'}}
                   }

    # load dataset
    dataset = PygNodePropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    data = SIGN(args.num_layers)(dataset[0])  # This might take a while.

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    #define data and label dict
    data_dict = {}
    label_dict = {}

    xs = [data.x] + [data[f'x{i}'] for i in range(1, args.num_layers + 1)]
    data_dict['train'] = xs_train = [x[split_idx['train']].to(device) for x in xs]
    data_dict['valid'] = xs_valid = [x[split_idx['valid']].to(device) for x in xs]
    data_dict['test'] = xs_test = [x[split_idx['test']].to(device) for x in xs]

    label_dict['train'] = y_train_true = data.y[split_idx['train']].to(device)
    label_dict['valid'] = y_valid_true = data.y[split_idx['valid']].to(device)
    label_dict['test'] = y_test_true = data.y[split_idx['test']].to(device)

    model = MLP(data.x.size(-1), args.hidden_channels, dataset.num_classes, args.num_layers,
                args.dropout).to(device)

    logger = Logger(args.runs, args)

    # calculate weights for loss function and fetch unique labels
    weights, unique_labels = calculate_weight_vector(data.y, return_unique_labels=True)
    # put weights tensor to defined device
    weights = weights.to(device)

    ####################################################################################################################
    #                               Directory Generateion Part1
    ####################################################################################################################
    # determine algorithm logged algorithm executions and define execution directory
    ex_prefix = 'ex'
    # check if logging root directory exists
    # define log dir path
    log_dir = os.path.join(args.log_dir, args.dataset, alg_name)
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

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.test_only:
            metric_res_dict = extended_test(model, data_dict, label_dict, device, metric_dict, unique_labels)
            result = [list(metric_res_dict['train'].values())[0], list(metric_res_dict['valid'].values())[0],
                        list(metric_res_dict['test'].values())[0]]

            train_acc, valid_acc, test_acc = result
            print(
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}% '
            f'Test: {100 * test_acc:.2f}%')

            exit()
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, xs_train, y_train_true, optimizer, device, weights)
            # tensorboard: log train loss
            writer.add_scalar("Loss/train", loss, epoch)
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}')

            if epoch > 9 and epoch % args.eval_steps == 0:

                # call extended test function
                metric_res_dict = extended_test(model, data_dict, label_dict, device, metric_dict, unique_labels)
                # extract results of one metric for command line outputs
                result = [list(metric_res_dict['train'].values())[0], list(metric_res_dict['valid'].values())[0],
                          list(metric_res_dict['test'].values())[0]]

                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result

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

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')
                if epoch > 70 and n_eval_no_best * args.eval_steps > args.n_stop_train:
                    break

        logger.print_statistics(run)
        best_metric_dict.update({'loss': loss, 'trained_epochs': epoch,
                                 'diff_train_valid_acc': float(train_acc - valid_acc),
                                 'execution_time': time.clock() - start_time})
        hparams_metric_dict = {f'hparam/{key}': value for key, value in best_metric_dict.items()}
        writer.add_hparams(
            vars(args),
            hparams_metric_dict)
        writer.flush()
        writer.close()
    logger.print_statistics()


if __name__ == "__main__":
    main()
