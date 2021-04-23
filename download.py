#!/usr/bin/env python3
import argparse
import torchvision.datasets as dset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True, choices=['cifar10',  'mnist'])
parser.add_argument('--dataroot', required=True, help='Path to store the dataset')
argv = parser.parse_args()

if argv.dataset == 'cifar10':
    dset.CIFAR10(root=argv.dataroot, download=True)
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=argv.dataroot, download=True)

print(f"Dataset saved to {argv.dataroot}.")
