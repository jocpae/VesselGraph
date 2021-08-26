export CUDA_DEVICE_ORDER=PCI_BUS_ID

sh final_gnn_gcn_parallelize.sh &
sh final_gnn_sage_parallelize.sh &
wait
sh final_gnn_gcn_emb_parallelize.sh & 
sh final_gnn_sage_emb_parallelize.sh &

