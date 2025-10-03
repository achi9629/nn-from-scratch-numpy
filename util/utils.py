import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import datasets
import random, argparse, datetime

from util.config import update_config, config

def create_folder(config):
    os.makedirs("experiments", exist_ok=True)
    folder_name = f"MLP_{config['seed']}_{config['batch_size']}"

    exp_dir = os.path.join("experiments", folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    df = pd.DataFrame(columns = ['Epoch', 'Loss_train', 'Loss_test', 'Accuracy_train', 'Accuracy_test'])

    return df, "experiments", folder_name

def parse_args():
    parser = argparse.ArgumentParser(description='MLP Configuration File')
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    return args

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def standardization(x):
    mean = x.mean()
    std = x.std()
    x_new = (x - mean)/std
    return x_new

def Accuracy(prob, target):
    return (prob == target).mean()

def mnist_data():
    train_data = datasets.MNIST(root="./data", train=True, download=True)
    test_data  = datasets.MNIST(root="./data", train=False, download=True)

    images_train = []
    labels_train = []
    for i in tqdm(range(len(train_data))):
        images_train.append(np.array(train_data[i][0])/255.)
        labels_train.append(train_data[i][1])
    images_train = np.stack(images_train)
    labels_train = np.array(labels_train)
    images_train_std = standardization(images_train.copy())
    
    ## Valid Data
    images_valid = []
    labels_valid = []
    for i in tqdm(range(len(test_data))):
        images_valid.append(np.array(test_data[i][0])/255.)
        labels_valid.append(test_data[i][1])
    images_valid = np.stack(images_valid)
    labels_valid = np.array(labels_valid)
    images_valid_std = standardization(images_valid.copy())
    
    return images_train_std, labels_train, images_valid_std, labels_valid