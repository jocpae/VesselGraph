import os
import numpy as np

alg_name = 'node2vec'
parallel_starts = 16

available_gpus = ['0,8', '8,0']
# args dict
args_dict = {'dataset': 'node_vessap_roi3_pb_minRadiusAvg',
             'log_dir': '../tensorboard_logs',
             'device': 1,
             'embedding_dim': 128,
             'walk_length': 40,
             'context_size': 20,
             'walks_per_node': 10,
             'batch_size': 256,
             'lr': 1.e-02,
             'epochs': 2,
             'log_steps': 1}

# define hyperparameter for grid search
hparam_dict = {
    'lr': [1.e-05, 1.e-03, 1e-2],  # np.logspace(-6, -3, 10),
    'walk_length': [16, 40],
    'context_size': [8, 15],
    'walks_per_node': [4, 10],
    'embedding_dim': [16, 128],
    'batch_size': [8, 64, 256],
    'epochs': [1, 4, 32]
}

####################################################################################################################
#                               Directory Generateion Part1
####################################################################################################################
# determine algorithm logged algorithm executions and define execution directory
ex_prefix = 'ex'
# check if logging root directory exists
# define log dir path
log_dir = os.path.join(args_dict['log_dir'], args_dict['dataset'], alg_name)
if not os.path.exists(log_dir):
    # if not generate it
    os.makedirs(log_dir)

# fetch list of directories in logging root directory
dir_entries = os.listdir(log_dir)
# generate list of sub directories with the defined subdirectory prefix
prev_logs = [name for name in dir_entries if name[:2] == ex_prefix]
# check if there exist subdirectories with the defined prefix
if len(prev_logs) > 0:
    # if so: set the index to the next higher number
    prev_logs.sort(reverse=True)
    args_dict['ex'] = int(prev_logs[0][-2:]) + 1
else:
    # if not: set index to 0
    args_dict['ex'] = 0

## initialize grid search
# initialize a index list of the current hyperparameter value for grid search
cur_idx_list = [0] * len(hparam_dict.keys())
# calculate number of hyperparameter combinations resulting from the defined values under test
comb = np.product([len(v) for v in hparam_dict.values()])
# add number of parameter combinations that are tested to parameter dictionary
args_dict['n_par_combs'] = int(comb)
# Output Status Message
print(f"Grid Search runs train loop for {comb} different hyperparameter combinations")

scripts_started = 0
with open('parallelize_node2vec.sh', 'w+') as f:
    f.write(f'export CUDA_DEVICE_ORDER=PCI_BUS_ID \n')
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
        param_str = ''
        for key, value in args_dict.items():
            param_str = f'{param_str} --{key} {value}'
        f.write(f'CUDA_VISIBLE_DEVICES={available_gpus[i % len(available_gpus)]} python node2vec.py{param_str} &\n')
        if scripts_started == parallel_starts:
            f.write('wait\necho \"wait for processes\"')
            scripts_started = 0
    f.write(f'finished hparam search')
