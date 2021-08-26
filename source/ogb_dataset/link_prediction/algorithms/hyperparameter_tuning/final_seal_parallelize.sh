CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 0 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 1 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 2 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 3 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 4 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 5 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 6 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 7 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 8 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 9 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 10 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 11 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 12 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 13 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 14 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 15 --num_hops 1 --model DGCNN &
wait
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 16 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 17 --num_hops 1 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 18 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 19 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 20 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 21 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 22 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 23 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 24 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 25 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 26 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 27 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 28 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 29 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 30 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 31 --num_hops 2 --model DGCNN &
wait
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 32 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 33 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 34 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 35 --num_hops 2 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 36 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 37 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 38 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 39 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 40 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 41 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 42 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 43 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 44 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 45 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 46 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 47 --num_hops 3 --model DGCNN &
wait
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 48 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 49 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 50 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 51 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 52 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 53 --num_hops 3 --model DGCNN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 54 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 55 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 56 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 57 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 58 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 59 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 60 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 61 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 62 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 63 --num_hops 1 --model GCN &
wait
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 64 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 65 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 66 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 67 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 68 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 69 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 70 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 71 --num_hops 1 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 72 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 73 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 74 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 75 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 76 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 77 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 78 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 79 --num_hops 2 --model GCN &
wait
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 80 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 81 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 82 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 83 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 84 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 85 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 86 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 87 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 88 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 89 --num_hops 2 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 90 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 91 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 92 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 93 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 94 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 32 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 95 --num_hops 3 --model GCN &
wait
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 96 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 97 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 98 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 99 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 100 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 64 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 101 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 102 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 2 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 103 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 104 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 3 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 105 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=7,1 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 0.0001 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 106 --num_hops 3 --model GCN &
CUDA_VISIBLE_DEVICES=1,7 python seal_link_pred_hyper.py --splitting_strategy spatial --use_feature  --log_steps 1 --num_layers 4 --hidden_channels 128 --batch_size 32 --lr 1e-05 --epochs 100 --eval_steps 10 --runs 1 --log_dir final_log_seal/run00 --n_par_combs 108 --curr_param_idx 107 --num_hops 3 --model GCN &
