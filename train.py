from __future__ import print_function
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, Sequential
from keras.datasets import fashion_mnist
import pdb

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path, mkdir, getcwd
import multiprocessing as mp
from multiprocessing import Pool

import vggish_params
from networks import get_vggish, get_n1
from vggish_input import wavfile_to_examples
from keras.callbacks import ModelCheckpoint, ProgbarLogger, TensorBoard
from keras.optimizers import Adam, SGD, Adagrad
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import GridSearchCV


def pool_map(func, args, n_proc=2):
    p = Pool(n_proc)
    ret = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            ret.append(res)
    pbar.close()
    p.close()
    p.join()
    return ret


def file_parts(fn):
    base_name, ext = path.splitext(path.basename(fn))
    return path.dirname(fn), base_name, ext


def extract_features(fns):
    feats = pool_map(wavfile_to_examples, fns, n_proc=mp.cpu_count())
    return feats


def get_data(fn):
    meta_fn = "{}.meta".format(fn)
    labels_fn = "{}.labels".format(fn)
    df = pd.read_csv(meta_fn)
    features = extract_features(df["fname"].values)
    labels = df["label"].values
    n_labels = len(df['label'].unique())
    final_features = []
    final_labels = []
    i = 0
    for example in features:
        for j in range(example.shape[0]):
            final_features.append(example[j])
            final_labels.append(labels[i])
        i += 1
    return final_features, final_labels


class Experiment(object):

    DEFAULTS = {
        "n_epochs": 10,
        "batch_size": 500,
        "checkpoint_period": 100,
        "lr": 0.001,
        "momentum": 0.5
    }

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", Experiment.DEFAULTS["batch_size"])
        self.n_epochs = kwargs.get("n_epochs", Experiment.DEFAULTS["n_epochs"])
        self.checkpoint_period = kwargs.get("checkpoint_period", Experiment.DEFAULTS["checkpoint_period"])
        self.lr = kwargs.get("lr", Experiment.DEFAULTS["lr"])
        self.momentum = kwargs.get("momentum", Experiment.DEFAULTS["momentum"])
        self.checkpoint = ModelCheckpoint(
            "weights.{epoch:02d}-{loss:.2f}.hdf5",
            monitor="loss",
            save_best_only=True,
            period=self.checkpoint_period
        )

        # ensure the checkpoints directory exists
        __checkpoint_dir = "checkpoints"
        if not path.exists(__checkpoint_dir):
            mkdir(__checkpoint_dir)

    def run(self):
        # get the model
        model = get_n1()
        #model.load_weights("vggish_weights.ckpt")
        model.add(Dense(41, activation='softmax', name='out'))
        #model.add(Dense(10, activation='softmax', name='out'))
        #model.load_weights("weights.18-0.16.hdf5")

        optimizer = SGD(lr=self.lr, momentum=self.momentum)

        progress_logger = ProgbarLogger()
        tensorboard = TensorBoard(log_dir=getcwd())
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['categorical_accuracy']
        )

        # load the data
        #print("Extracting features for train data.")
        #X_train, y_train, fn_train = get_data("strong")
        #binarizer = LabelBinarizer()
        #binarizer.fit(y_train)
        #y_train = binarizer.transform(y_train)
        #X_train = np.array(X_train).reshape(len(X_train), 96, 64)
        #np.savez_compressed("strong.npz", X=X_train, y=y_train, fn=fn_train)

        #print("Extracting features for test data.")
        #X_val, y_val, fn_val = get_data("weak")
        #y_val = binarizer.transform(y_val)
        #X_train = np.array(X_train).reshape(len(X_train), 96, 64, 1)
        #X_val = np.array(X_val).reshape(len(X_val), 96, 64, 1)
        #np.savez_compressed("weak.npz", X=X_val, y=y_val, fn=fn_val)

        print("Loading train data.")
        # data generated from vggish_input.py
        data = np.load("strong.npz")
        X_train, y_train, fn_train = data["X"], data["y"], data["fn"]
        print("Centering and normalizing train data.")
        X_train = X_train - X_train.mean(axis=0)
        x_min = X_train.min()
        X_train = (X_train - x_min) / (X_train.max() - x_min)
        print(X_train.min(), X_train.max(), X_train.std())

        #(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        #X_train = X_train.reshape((len(X_train), X_train.shape[1], X_train.shape[2], 1))
        #import pdb; pdb.set_trace()

        #data = map(lambda fn: np.load(path.join('first_windows', fn)), os.listdir('first_windows'))
        #meta = pd.read_csv('all.meta')
        #fns = pd.DataFrame(fn_train, columns=['fn'])
        #fns['id'] = fns['fn'].map(lambda fn: path.basename(fn).split('.')[0])
        #y_train = LabelBinarizer().fit_transform(fns.merge(meta, on='id')['label'].values)

        #print("Loading test data.")
        #data = np.load("weak.npz")
        #X_val, y_val, fn_val = data["X"], data["y"], data["fn"]
        #print("Done")

        #print("Loading train data.")
        #data = np.load("train.npz")
        #X_train, y_train = data["X_train"], data["y_train"]
        #print("Loading test data.")
        #data = np.load("val.npz")
        #X_val, y_val = data["X_val"], data["y_val"]
        #print("Done")

        # train
        model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=[self.checkpoint, progress_logger],
            verbose=2
        )

        #y_pred_proba = model.predict_proba(X_val, batch_size=self.batch_size)

        #def bn(fn):
        #    return fn.split('/')[6].split('.')[0]

        #labels_mapping = {
        #    bn(f): y
        #    for f, y in zip(fn_val, y_val)
        #}
        #labels_df = pd.DataFrame.from_dict(labels_mapping).transpose()
        #np.savez_compressed("preds_mapping.npz",
        #                    fn=labels_df.index.values,
        #                    y_pred=y_pred_proba.argmax(axis=1),
        #                    y_pred_proba=y_pred_proba,
        #                    y=y_val)


def main():
    if __name__ == '__main__':
        print("**********************")
        Experiment(lr=1e-4, momentum=0.9, batch_size=10, n_epochs=250, checkpoint_period=50).run()
        #print("**********************")
        #Experiment(lr=0.0001, momentum=0.9, batch_size=10, n_epochs=250, checkpoint_period=50).run()
        #print("**********************")
        #Experiment(lr=0.0001, momentum=0.9, batch_size=20, n_epochs=250, checkpoint_period=50).run()
        #print("**********************")
        #Experiment(lr=0.0001, momentum=0.9, batch_size=10, n_epochs=250, checkpoint_period=50).run()
        #print("**********************")
        #Experiment(lr=0.0001, momentum=0.9, batch_size=10, n_epochs=250, checkpoint_period=50).run()
        #print("**********************")
        #Experiment(lr=0.00001, momentum=0.9, batch_size=20, n_epochs=250, checkpoint_period=50).run()


main()
