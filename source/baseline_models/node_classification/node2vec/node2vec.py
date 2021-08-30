alg_name = 'node2vec'

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
from torch_geometric.nn import Node2Vec

from ogb.nodeproppred import PygNodePropPredDataset

from torch.utils.tensorboard import SummaryWriter


def save_embedding(model, file_name='embedding.pt'):
    torch.save(model.embedding.weight.data.cpu(), file_name)


def main():
    parser = argparse.ArgumentParser(description='ogbn-italo (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=15)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='../tensorboard_logs')
    parser.add_argument('--target_dir', type=str, default='embeddings')
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--n_par_combs', type=int, default=0)
    parser.add_argument('--curr_param_idx', type=int, default=0)
    parser.add_argument('--n_stop_train', type=int, default=160)
    parser.add_argument('--ex', type=int, default=-1)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name=f'ogbn-{args.ds_name}')
    data = dataset[0]
    if not os.path.exists(args.target_dir):
        # if not generate it
        os.makedirs(args.target_dir)
    file_name = os.path.join(args.target_dir, f'{args.ds_name}_embedding.pt')
    model = Node2Vec(data.edge_index, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    model.train()

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
    log_dir = os.path.join(log_dir, f'{ex_prefix}{curr_dir_idx:0>2}')

    # define logging sub_directory name suffix for summary writer (index of current parameter_combination
    if args.n_par_combs > 1:
        log_dir = os.path.join(log_dir, f'{args.curr_param_idx:0>3}of{args.n_par_combs}')
    ####################################################################################################################
    ####################################################################################################################

    # instantiate tensorboard writer
    writer = SummaryWriter(log_dir)

    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            writer.add_scalar("Loss", loss, epoch)
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 10000 == 0:  # Save model every 100 steps.
                save_embedding(model, file_name)
        save_embedding(model, file_name)
    writer.add_hparams(
        vars(args),
        {'hparam/loss': loss})
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
