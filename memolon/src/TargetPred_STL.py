#!/usr/bin/env python

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memolon.src import utils
from memolon.src import model
import memolon.src.constants as cs

language_table = utils.get_language_table()
iso_codes = language_table.index.tolist()


##### creating TargetPred lexicons (STL, so single-task learning each emotional variable separately)

for iso in iso_codes:
    print('creating lexicon for {} ...'.format(iso))

    # load embeddings
    embs = utils.load_vectors(iso=iso)

    # load TargetMT for training
    target_mt = utils.get_TargetMT(iso=iso, split="full")
    train = utils.get_TargetMT(iso=iso, split="train")
    dev = utils.get_TargetMT(iso=iso, split="dev")

    # creating empty TargetPred: list of words to predict values for (entries in TargetMT + words in embeddings + <UNK>)
    total_words = target_mt.index.tolist() + list(embs.keys())
    total_words.append('<UNK>')

    target_pred = pd.DataFrame(0, index=total_words,
                           columns=['valence', 'arousal', 'dominance', 'joy', 'anger', 'sadness', 'fear', 'disgust'])
    target_pred.index.name = 'word'

    # features = embeddings of translated entries; labels = emotion values

    X_train = [utils.get_features(entry, embs) for entry in train.index]
    X_dev = [utils.get_features(entry, embs) for entry in dev.index]

    X_train = torch.Tensor(np.array(X_train))
    y_train = np.column_stack((train.valence, train.arousal, train.dominance,
                               train.joy, train.anger, train.sadness, train.fear, train.disgust))
    y_train = torch.Tensor(np.array(y_train))

    X_dev = torch.Tensor(np.array(X_dev))
    y_dev = np.column_stack((dev.valence, dev.arousal, dev.dominance,
                             dev.joy, dev.anger, dev.sadness, dev.fear, dev.disgust))
    y_dev = torch.Tensor(np.array(y_dev))


    # putting features and labels together to use DataLoader
    train_list, dev_list = [], []
    for i in range(len(X_train)):
        train_list.append([X_train[i], y_train[i]])
    for j in range(len(X_dev)):
        dev_list.append([X_dev[j], y_dev[j]])

    # create DataLoaders (batches of features and labels)
    b_size = 128
    train_set = torch.utils.data.DataLoader(train_list, batch_size=b_size, shuffle=True)
    dev_set = torch.utils.data.DataLoader(dev_list, batch_size=b_size, shuffle=True)

    ########## Training model ############

    # all 8 variables are trained separately
    variables = ['valence', 'arousal', 'dominance', 'joy', 'anger', 'sadness', 'fear', 'disgust']

    for n_var, var in enumerate(variables):

        net = model.Net(1)

        # loss and optimizer
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        print('beginning training variable {} (so variable nr. {}) for {} ...'.format(var, n_var, iso))

        for epoch in range(168):
            print('epoch: {}'.format(epoch + 1))

            # training on train_set
            running_train_loss = 0.0
            i = 0

            for batch in train_set:
                features, labels = batch  # warning: labels are all dimensions (VAD + JASFD)
                labels = labels[:, n_var]  # just use the one variable

                net.zero_grad()
                net.train()  # apply dropout
                outputs = net(features)
                outputs = outputs.squeeze()

                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                i += 1

            avg_train_loss = running_train_loss / i
            print('training loss:\t {}'.format(avg_train_loss))

            # validation on dev_set
            running_val_loss = 0.0
            i = 0

            for batch in dev_set:
                features, labels = batch  # warning: labels are all dimensions
                labels = labels[:, n_var]

                net.zero_grad()
                net.eval()  # so that dropout is NOT applied during evaluation/inference
                with torch.no_grad():  # tracking gradients not necessary for inference
                    outputs = net(features)
                    outputs = outputs.squeeze()
                    loss = loss_function(outputs, labels)
                running_val_loss += loss.item()
                i += 1

            avg_val_loss = running_val_loss / i
            print('validation loss:\t {}'.format(avg_val_loss))

        ########## Using trained model to predict #############

        # get features for total of words
        X_total = [utils.get_features(entry, embs) for entry in total_words]
        X_total = torch.Tensor(np.array(X_total))

        net.eval()  # so that dropout is NOT applied during inference
        with torch.no_grad():  # tracking gradients not necessary for inference
            predictions = net(X_total)
            predictions = predictions.numpy()
            predictions = np.around(predictions, decimals=2)

        #print("Shape of predictions:", predictions.shape)

        # write predictions (one column) into initially empty TargetPred
        target_pred[var] = predictions

    # postprocessing lexicon (sorting and removing duplicates)
    sorted = utils.postprocess_lexicon(iso, target_pred)

    # save TargetPred
    print(sorted.head())
    print(sorted.shape)
    sorted.to_csv(cs.TARGETPRED_STL / '{}.tsv'.format(iso), sep='\t', index=True, encoding='utf-8')
    print('done with TargetPred for {}'.format(iso))

