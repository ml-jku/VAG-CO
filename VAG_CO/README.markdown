# Running VAG-CO experiments

## Training

After you created the datasets you can train your model with:

e.g.
```train
python start_parallel_runs.py --IsingMode RB_iid_100 --temps 0.05 --N_anneal 4000 --GPUs 0 --time_horizon 20 --batch_epochs 2 --AnnealSchedule cosine_frac --num_hidden_neurons 40 --mini_Sb 10 --mini_Nb 10 --mini_Hb 10 --EnergyFunction MVC --n_val_workers 30 --encode_node_features 40 --encode_edge_features 30 --lr 0.0005 --message_nh 30 --seed 123 --n_GNN_layers 3 --n_sample_spins 5 --project_name "myfirstrun"
```

You can find other parser_args that were used in the paper either in parser_MVC.txt or parser_MIS.txt.
The list of all parser_args will be added soon.

## Evaluation

You can evaluate on a dataset by running evaluate.py

e.g.

```train
python evaluate.py --GPU 0 --batchsize 2 --Ns 8 --sample_mode OG
```

This code will run evaluation on the RRG-100 MIS dataset and calculate an average APR.
If you want to evaluate on another dataset you will have to change the "overall_path" in evaluate.py to a path that contains the config and model weights of a model that is trained on another dataset.

## Pre-trained Models

A pretrained model on the RB_200 dataset will be added soon.
See Evaluation. 

## Results

see Paper.
