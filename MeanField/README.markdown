# Running expermiments with EGN or MFA

## Training
Runs can be started with using ```argparse_ray_main.py```.

Exampel of two parallel runs at GPU 2 and 3 with different temperatures.
```--realaxed``` means that EGN is used. If ```--no-relaxed``` is used MFA will be used.
```
 python argparse_ray_main.py --lrs 0.00005 --relaxed --GPUs 2 3 --n_GNN_layers 8 --temps 0.25 0.5 0. --IsingMode RB_iid_200 --EnergyFunction MIS
```

## Evaluation

Evaluation can be run in ```ConditionalExpectation.py```.


Example code is given here:

```
    from ConditionalExpectation import ConditionalExpectation
    
    device = 0 ### SPecify GPU 

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    print(f"GPU: {device}")

    wandb_id = "3iut7t6v" ### Specify wandb_run_id
    CE = ConditionalExpectation(wandb_id=wandb_id, n_different_random_node_features=8)
    CE.init_dataset(dataset_name=None)
    CE.run(p=None)
    print('\n###\n')

```

