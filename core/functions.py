import numpy as np
from tqdm import tqdm

from util.utils import Accuracy

def train(images_train_std, labels_train, model, categorical_loss, batch_size):
    loss_track = []
    pred_track = []
    gt_track = []
    n_batches = images_train_std.shape[0]//batch_size + 1
    for batch in tqdm(range(n_batches)):
        inp = images_train_std[batch_size*batch:min(batch_size*(batch+1), images_train_std.shape[0])]
        Y   = labels_train[batch_size*batch:min(batch_size*(batch+1), images_train_std.shape[0])]
        output = model.forward(inp)
        loss = categorical_loss.forward(output, Y)
        model.backward(categorical_loss.backward())
        model.step()
        loss_track.append(loss)
        pred_track.append(output.argmax(1))
        gt_track.append(Y)
    
    pred_track = np.hstack(pred_track)
    gt_track = np.hstack(gt_track)
    loss_track = np.hstack(loss_track)
    # print('Train Loss, Accuracy: ', round(loss_track.mean(), 4), round(Accuracy(pred_track, gt_track), 4))
    return round(loss_track.mean(), 4), round(Accuracy(pred_track, gt_track), 4)

def valid(images_valid_std, labels_valid, model, categorical_loss, batch_size):
    loss_track = []
    pred_track = []
    gt_track = []
    n_batches = images_valid_std.shape[0]//batch_size + 1
    for batch in tqdm(range(n_batches)):
        inp = images_valid_std[batch_size*batch:min(batch_size*(batch+1), images_valid_std.shape[0])]
        Y   = labels_valid[batch_size*batch:min(batch_size*(batch+1), images_valid_std.shape[0])]
        output = model.forward(inp)
        loss = categorical_loss.forward(output, Y)
        loss_track.append(loss)
        pred_track.append(output.argmax(1))
        gt_track.append(Y)
    pred_track = np.hstack(pred_track)
    gt_track = np.hstack(gt_track)
    loss_track = np.hstack(loss_track)
    # print('Valid Loss, Accuracy: ', round(loss_track.mean(), 4), round(Accuracy(pred_track, gt_track), 4))
    return round(loss_track.mean(), 4), round(Accuracy(pred_track, gt_track), 4), pred_track, gt_track