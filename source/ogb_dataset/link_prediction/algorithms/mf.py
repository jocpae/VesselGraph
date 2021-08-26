import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../../../')
sys.path.insert(0, '../../../ogb')
import os

import argparse
import time
from shutil import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from sklearn.metrics import roc_auc_score

from logger import Logger

# from SEAL
def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)

    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_train_pred,
    })['mrr_list'].mean().item()
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr,valid_mrr, test_mrr)
    
    return results


# from SEAL
def evaluate_auc(train_pred,train_true,val_pred, val_true, test_pred, test_true):
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (train_auc,valid_auc, test_auc)

    return results



class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.norms = torch.nn.ModuleList()
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin, norm in zip(self.lins[:-1], self.norms):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(x, predictor, split_edge, optimizer, batch_size,splitting_strategy):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    if splitting_strategy == 'spatial':

        neg_train_edge = split_edge['train']['edge_neg'].to(x.device) # modified

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()
        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        if splitting_strategy == 'random':

            # Just do some trivial random sampling.
            edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                                device=x.device)
            neg_out = predictor(x[edge[0]], x[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        elif splitting_strategy == 'spatial':
            edge = neg_train_edge[perm].t()
            neg_out = predictor(x[edge[0]], x[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        else:
            raise ValueError("Splitting Strategy not defined!")


        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(x, predictor, split_edge, evaluator, batch_size,eval_metric):
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    if eval_metric == 'mrr':
        neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
        neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
        neg_valid_pred = neg_valid_pred.view(pos_valid_pred.shape[0], -1)
        result = evaluate_mrr(evaluator, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    elif eval_metric == 'auc':
        train_pred = torch.cat([pos_train_pred, neg_train_pred])
        train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                              torch.zeros(neg_train_pred.size(0), dtype=int)])
        val_pred = torch.cat([pos_valid_pred, neg_valid_pred])
        val_true = torch.cat([torch.ones(pos_valid_pred.size(0), dtype=int), 
                              torch.zeros(neg_valid_pred.size(0), dtype=int)])
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                              torch.zeros(neg_test_pred.size(0), dtype=int)])
        result = evaluate_auc(train_pred,train_true,val_pred, val_true, test_pred, test_true)

    else:
        result = None
        raise ValueError('Evaluation Metric {eval_metric} not implemented for this dataset.')
        
    
    return result

def main():
    parser = argparse.ArgumentParser(description='OGBL MF algorithm')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0001) #0.005
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--eval_metric', type=str, default='auc')
    parser.add_argument('--dataset', type=str,default='ogbl_link_vessap_roi3_spatial_no_edge_attr')
    parser.add_argument('--splitting_strategy',type=str,default='spatial')

    # Log settings
    parser.add_argument('--save_appendix', type=str, default='', 
                        help="an appendix to the save directory")

    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f'Running on {args.dataset}')
    print(f'Utilizing evaluation metric {args.eval_metric}')

    dataset = PygLinkPropPredDataset(name=args.dataset)

    #  from Muhan Zhang's OGB SEAL repository

    if args.save_appendix == '':
        args.save_appendix = '_mf_'+ time.strftime("%Y%m%d%H%M%S")
 
    args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
    print('Results will be saved in ' + args.res_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir) 
        # Backup python files.
    copy('mlp.py', args.res_dir)
    log_file = os.path.join(args.res_dir, 'log.txt')

    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    with open(log_file, 'a') as f:
        f.write('\n' + cmd_input)


    split_edge = dataset.get_edge_split()
    data = dataset[0]

    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name=args.dataset)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        emb.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(emb.parameters()) + list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(emb.weight, predictor, split_edge, optimizer,
                         args.batch_size,args.splitting_strategy)

            if epoch % args.eval_steps == 0:
                results = test(emb.weight, predictor, split_edge, evaluator,
                               args.batch_size,args.eval_metric)

                for key, result in results.items():
                    train_res, valid_res, test_res = result
                    print(test_res, valid_res, test_res)
                
                    #print(len(result))
                    logger.add_result(run, result)

                    if epoch % args.log_steps == 0:
                        for key, result in results.items():
                            train_res, valid_res, test_res = result
                            print(key)
                            log_text = (f'Run: {run + 1:02d}, ' +
                                f'Epoch: {epoch:02d}, ' +
                                f'Loss: {loss:.4f}, ' +
                                f'Train: {100 * train_res:.2f}%, ' +
                                f'Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')

                            print(log_text)
                            with open(log_file, 'a') as f:
                                print(log_text, file=f)
                            
        logger.print_statistics(run)
        with open(log_file, 'a') as f:
            logger.print_statistics(run,f=f)

    logger.print_statistics()
    with open(log_file, 'a') as f:
        logger.print_statistics(f=f)


if __name__ == "__main__":
    main()