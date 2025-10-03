import os
import numpy as np
import pandas as pd

from util.loss import NLL
from model.model import MLP
from util.config import config
from core.functions import train, valid
from util.optimizer import Optimizer, get_optim_hyperparam
from util.vizualization import plot_training_curves, plot_classification_metrics
from util.utils import set_seed, standardization, Accuracy, mnist_data, parse_args, create_folder

def main():
    args = parse_args()
    df, out_dir, exp_dir = create_folder(config)
    images_train_std, labels_train, images_valid_std, labels_valid = mnist_data()

    batch_size = config.batch_size
    epochs = config.epochs
    set_seed(config.seed)
    
    categorical_loss = NLL()
    lr, momemt1, momemt2 = get_optim_hyperparam(config.optimizer.otype)
    optim = Optimizer(otype = config.optimizer.otype, lr =lr, momentum1 = momemt1, momentum2 = momemt2)
    model = MLP(config.layer_dims, optim)

    for epoch in range(epochs):
        print('*********************** Epoch: {} ****************************************'.format(epoch))
        loss_train, accuracy_train = train(images_train_std, labels_train, model, categorical_loss, batch_size)
        loss_valid, accuracy_valid, _, _ = valid(images_valid_std, labels_valid, model, categorical_loss, batch_size)

        df.loc[epoch, ['Epoch']] = epoch
        df.loc[epoch, ['Loss_train']] = loss_train
        df.loc[epoch, ['Loss_test']] = loss_valid
        df.loc[epoch, ['Accuracy_train']] = accuracy_train
        df.loc[epoch, ['Accuracy_test']] = accuracy_valid
        df.to_csv(os.path.join(out_dir, exp_dir, 'train_stats_' + config.optimizer.otype + '.csv'), index=False)
    plot_training_curves(config, df, os.path.join(out_dir, exp_dir, 'training_curves_' + config.optimizer.otype + '.png'))
    
    loss_valid, accuracy_valid, pred_track, gt_track = valid(images_valid_std, labels_valid, model, categorical_loss, batch_size)
    
    dictionary = {'Pred': pred_track, 'GT': gt_track}
    df_valid = pd.DataFrame(dictionary)
    plot_classification_metrics(config, df_valid, os.path.join(out_dir, exp_dir, 'metrics_' + config.optimizer.otype + '.png'))
    df_valid.to_csv(os.path.join(out_dir, exp_dir, 'test_stats_' + config.optimizer.otype + '.csv'), index=False)
    

if __name__=='__main__':
    main()