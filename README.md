# GNN-Test-CodeDepot

A Graph Neural Network implementation.

# Table of Contents
1. [Installation](#installation)
2. [Test-the-model](#Test-the-model)
3. [Watching-the-GPU](#watching-the-GPU)

# Installation
First, we need to recreate a conda enviroment to run the codes: 
```bash
conda env create -f graph_env.yml
```

# Test-the-model
First, we can test the gnn model using the `gnn_test.py` file. Make sure to activate the conda enviroment to run this code:

```bash
conda activate phi3

python3 sanity_test.py
```

# Watching the GPU
To monitor the gpu during training, at another end, I used the command:

```bash
watch -n1 nvidia-smi
```
This command will update the GPU information every 1 second, allowing us to monitor how much of the GPU is being used during script execution.



