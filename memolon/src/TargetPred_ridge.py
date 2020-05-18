#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from memolon.src import utils
import memolon.src.constants as cs

language_table = utils.get_language_table()
iso_codes = language_table.index.tolist()


##### creating TargetPred lexicons (ridge baseline)

for iso in iso_codes:
    print('creating lexicon for {} ...'.format(iso))

    # load embeddings
    embs = utils.load_vectors(iso=iso)

    # load TargetMT for training
    target_mt = utils.get_TargetMT(iso=iso, split="full")
    train = utils.get_TargetMT(iso=iso, split="train")

    X_train = [utils.get_features(entry, embs) for entry in train.index]
    X_train = np.array(X_train)
    y_train = np.column_stack((train.valence, train.arousal, train.dominance,
                               train.joy, train.anger, train.sadness, train.fear, train.disgust))

    # train model: Ridge regression
    ridge = Ridge()
    ridge.fit(X_train, y_train)

    # predicting for total words
    total_words = target_mt.index.tolist() + list(embs.keys())
    total_words.append('<UNK>')

    X_total = [utils.get_features(entry, embs) for entry in total_words]
    X_total = np.array(X_total)
    predictions = np.around(ridge.predict(X_total), decimals=2)

    # write predictions into pandas.DataFrame
    target_pred = pd.DataFrame(predictions, columns=['valence', 'arousal', 'dominance', 'joy', 'anger', 'sadness', 'fear', 'disgust'])
    target_pred.insert(loc=0, column='word', value=total_words)
    target_pred.set_index('word', inplace=True)

    # postprocessing lexicon (sorting and removing duplicates)
    sorted = utils.postprocess_lexicon(iso, target_pred)

    # save TargetPred
    print(sorted.head())
    print(sorted.shape)
    sorted.to_csv(cs.TARGETPRED_RIDGE / '{}.tsv'.format(iso), sep='\t', index=True, encoding='utf-8')
    print('done with TargetPred for {}'.format(iso))
