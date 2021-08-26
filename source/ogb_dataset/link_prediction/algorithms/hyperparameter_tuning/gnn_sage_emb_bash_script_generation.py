import os
import numpy as np

parallel_starts = 16

available_gpus = ['5,7', '7,5']

# args dict
args_dict = {'device': 6, # gpu
           'log_steps': 1,
           'num_layers': 3,
           'hidden_channels': 256,
           'dropout': 0.0,
           'batch_size': 64 * 1024,
           'lr': 1.e-03,
           'epochs': 1000,
           'eval_steps': 10,
           'runs': 1}

# define hyperparameter for grid search
hparam_dict = {
    'lr': [1.e-03, 1.e-04, 1.e-05],  # np.logspace(-6, -3, 10),
    'num_layers': [2, 3, 4],
    'hidden_channels': [128, 256, 512],
    'dropout':[0.0,0.2,0.5],
}
# define root directory for tensorboard logging data
log_root = 'final_log_gnn_sage_emb'
# define prefix of logging information subdirectories
log_dir_prefix = 'run'

# check if logging root directory exists
if not os.path.exists(log_root):
    # if not generate it
    os.makedirs(log_root)
# fetch list of directories in logging root directory
dir_entries = os.listdir(log_root)
# generate list of sub directories with the defined subdirectory prefix
prev_logs = [name for name in dir_entries if name[:3] == log_dir_prefix]
# check if there exist subdirectories with the defined prefix
if len(prev_logs) > 0:
    # if so: set the index to the next higher number
    prev_logs.sort(reverse=True)
    curr_dir_idx = int(prev_logs[0][-2:]) + 1
else:
    # if not: set index to 0
    curr_dir_idx = 0

# stitch resulting directory for tensorboard logging data
log_dir = os.path.join(log_root, f'{log_dir_prefix}{curr_dir_idx:0>2}')

args_dict['log_dir'] = log_dir

## initialize grid search
# initialize a index list of the current hyperparameter value for grid search
cur_idx_list = [0] * len(hparam_dict.keys())
# calculate number of hyperparameter combinations resulting from the defined values under test
comb = np.product([len(v) for v in hparam_dict.values()])
# add number of parameter combinations that are tested to parameter dictionary
args_dict['n_par_combs']  = int(comb)
# Output Status Message
print(f"Grid Search runs train loop for {comb} different hyperparameter combinations")

scripts_started = 0
with open('final_gnn_sage_emb_parallelize.sh','w+') as f:
    # loop over all possible hyperparameter combinations
    for i in range(comb):
        scripts_started += 1
        # update current parameter combination at parameter dictionary
        args_dict['curr_param_idx'] = i
        # select hyperparameters from the hyperparameter dict so that no combination occures two times
        mdl = 1
        for j, (key, value) in enumerate(hparam_dict.items()):
            if i % mdl == 0:
                args_dict[key] = value[cur_idx_list[j]]
                cur_idx_list[j] = (cur_idx_list[j] + 1) % len(value)
            mdl = mdl * len(value)
        # print(f"  - {i}. Combination: {args_dict}\n     Current Index List: {cur_idx_list}")
        param_str= ''
        for key, value in args_dict.items():
            param_str = f'{param_str} --{key} {value}'
        f.write(f'python gnn_hyper_final.py --use_sage --use_node_embedding {param_str} &\n')
        if scripts_started == parallel_starts:
            f.write('wait\n')
            scripts_started = 0






