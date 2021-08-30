import sys
sys.path.append('../../../')
import os
import argparse
import torch
from torch_geometric.nn import Node2Vec
from ogb.linkproppred import PygLinkPropPredDataset
from torch.utils.tensorboard import SummaryWriter

# limit load for the server
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=1

def save_embedding(dataset_name, model):
    embedding_name = 'node2vec_'+ dataset_name +'.pt'
    #torch.save(model.embedding.weight.data.cpu(), 'embedding.pt')
    torch.save(model.embedding.weight.data.cpu(), embedding_name)


def main():
    parser = argparse.ArgumentParser(description='OGBL (Node2Vec)')
    parser.add_argument('--dataset',type=str,default='ogbl-link_vessap_roi3_spatial_no_edge_attr')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--log_dir',type=str, default= "node2vec_log")
    parser.add_argument('--n_par_combs',type=int, default = 1) 
    parser.add_argument('--curr_param_idx', type=int, default = 1)

    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]

    model = Node2Vec(data.edge_index, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    writer = SummaryWriter(os.path.join(args.log_dir,f'{args.curr_param_idx}_of_{args.n_par_combs}'))

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            writer.add_scalar('loss', loss, epoch)
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')
           

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding(args.dataset,model)
        save_embedding(args.dataset,model)

    writer.add_hparams(
                vars(args),
                    {'hparam/loss':loss,})
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
