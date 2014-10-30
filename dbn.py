from nolearn.dbn import DBN
import argparse
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import sklearn
import itertools

from utils import unspeech_utils
from feature_gen import energy,windowed_fbank

from scipy.stats import itemfreq

from sklearn.cross_validation import StratifiedShuffleSplit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate FBANK features (=logarithmic mel frequency filter banks) for all supplied files in file list (txt).')
    parser.add_argument('-f', '--filelists-dir', dest="filelists", help='file list basedir', default='corpus/voxforge/', type=str)
    parser.add_argument('-v', dest='verbose', help='verbose output', action='store_true', default=False)
    parser.add_argument('-l', '--langs', dest='langs', help='comma seperated list of lang codes',type=str,default='de,fr')
    parser.add_argument('-m', '--max', dest='max_samples', help='max utterance samples per language',type=int,default=-1)
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)

    args = parser.parse_args()

    langs = (args.langs).split(',')
    lang2num = {}
    for i,lang in enumerate(langs):
        lang2num[lang] = i
    
    X_trains = []
    y_trains = []

    for lang in langs:
        ids = unspeech_utils.loadIdFile(args.filelists+'all_'+lang+'.txt',basedir=args.basedir)
        for myid in ids[:args.max_samples]:
            logspec_features = np.load(myid+'.logspec.npy')
            logspec_features_filtered = energy.filterSpec(logspec_features,1.2)
            feat = windowed_fbank.generate_feat(logspec_features_filtered,window_size=21,step_size=15)
            feat_len = feat.shape[0]
            feat_dim = feat.shape[1]
            labels = np.zeros(feat_len,dtype=int)
            labels[:] = lang2num[lang]
            X_trains.append(feat)
            y_trains.append(labels)

    X_all = np.asarray(X_trains)
    y_all = np.asarray(y_trains)

    y_all_flat = np.asarray([y[0] for y in y_all])

    #print X_all,y_all_flat

    sss = StratifiedShuffleSplit(y_all_flat, 1, test_size=0.1, random_state=0)
    
    train_index, test_index = iter(sss).next()

    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]

    X_train = np.vstack(X_train)
    #X_test = np.vstack(X_test)
    y_train = np.concatenate(y_train)
    #y_test = np.concatenate(y_test)

    y_test_flat = np.asarray([y[0] for y in y_test]) 
   # X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    #X_train, X_test, y_train, y_test = train_test_split(X_all,y_all, test_size=0.2)

    std_scale = StandardScaler(copy=False, with_mean=True, with_std=False).fit(X_train)
    #norm_scale = sklearn.preprocessing.Normalizer(norm='l1', copy=False).fit(X_train)
    
    X_train = std_scale.transform(X_train)

    print 'Done loading data, samplesize:', len(X_all)

    print 'Distribution of classes in complete data:'
    print itemfreq(y_all_flat)
    print y_test
    print 'Distribution of classes in test data:'
    print itemfreq(y_test_flat)

    #hid_layer_units = 2560 
    hid_layer_units = 1000

    clf = DBN([X_train.shape[1],hid_layer_units, hid_layer_units, len(langs)],
            #dropouts=100,
            learn_rates=0.001,
            learn_rates_pretrain=0.00001,
            minibatch_size=200,
            #learn_rate_decays=0.9,
            epochs=15,
            #use_re_lu=True,
            verbose=1)

    print 'fitting dbn...'

    clf.fit(X_train, y_train)

    print 'done!'

    y_pred = []
    for utterance,y in itertools.izip(X_test,y_test):
        utterance = std_scale.transform(utterance) 
        local_pred = clf.predict(utterance)
        voting = np.bincount(local_pred)
        #majority voting
        pred = np.argmax(voting)
        y_pred.append(pred)

    print "Accuracy:", accuracy_score(y_test_flat, y_pred)
    print "Classification report:"
    print classification_report(y_test_flat, y_pred)
    print "Confusion matrix:\n%s" % confusion_matrix(y_test_flat, y_pred)
