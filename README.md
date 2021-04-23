# Deep Convolution Generative Adversarial Networks

This repository is a distributed data parallel version of [dcgan.pytoch examples](https://github.com/pytorch/examples/tree/master/dcgan)

The example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

The implementation is very close to the Torch implementation [dcgan.torch](https://github.com/soumith/dcgan.torch)

After every 100 training iterations, the files `real_samples*.png` and `fake_samples*.png` are written to disk
with the samples from the generative model.

## Downloading Dataset

First download and extract the desired dataset. In this example we'll download and use CIFAR-10 dataset, which can be obtained using the commands below:

```console
mkdir cifar10 && cd cifar10
wget --no-check-certificate https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
cd ..
```

This will download and extract cifar-10 dataset to `./cifar-10-batches-py/`

## Installing and Running

1. Build the docker image from this repository using the command below:

```console
docker build -t dist_dcgan .
```

2. Let's start 2 terminals to test the network using data distributed parallelism.
On first terminal run:

```console
docker run --rm --network=host -v=$(pwd):/root dist_dcgan:latest python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.17.0.1" --master_port=1234 dist_dcgan.py --dataset cifar10 --dataroot ./cifar10
```

And in the second terminal, run:
```console
docker run --rm --network=host -v=$(pwd):/root dist_dcgan:latest python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="172.17.0.1" --master_port=1234 dist_dcgan.py --dataset cifar10 --dataroot ./cifar10
```

* The `-v` parameter is mapping the current directory to docker's `/root` directory, where data and the scripts resides.
* The `--nproc_per_node` is the number of processes to start in each node.
* The `--nnodes` parameter defines the number total of nodes (world_size).
* The `--node_rank` parameter defines the rank for this particular node. Each node must have an unique node_rank.
* The `--master_addr` and `-master_port` parameters is the address and port to the master node (rank 0), respectively. Check if docker's IP is visible otherwise the processing will be stuck and will never starts.
* The `--dataset` and `--dataroot` parameters indicates the dataset and where it is located, respectively.
* Use `--gpus=all` to docker command and `--cuda` to `dist_dcgan.py` to enable the use of GPUs. For multiple gpus in a single node, it is recommended to set `--nproc_per_node=1` (one process per gpu) and each for each process, use the `--local_rank` parameter to assign it a a specific GPU (or using CUDA_VISIBLE_DEVICES environment variable).  
