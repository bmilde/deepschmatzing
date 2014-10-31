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

#load and construct feature vectors for a single logspec file id
def loadIdFeat(myid,dtype, window_size, step_size):
    logspec_features = np.load(myid+'.logspec.npy')
    if(logspec_features.dtype != dtype):
        logspec_features = logspec_features.astype(dtype, copy=False)

    logspec_features_filtered = energy.filterSpec(logspec_features,1.2)
    feat = windowed_fbank.generate_feat(logspec_features_filtered,window_size=window_size,step_size=step_size)
    return feat

#load specgrams and generate windowed feature vectors
def loadTrainData(ids,classes,window_size,step_size):
    
    #iterate through all files and find out the space needed to store all data in memory
    required_shape = [0,0]
    for myid in ids:
        #do not load array into memory yet
        logspec_features_disk = np.load(myid+'.logspec.npy',mmap_mode='r')
        feat_gen_shape = windowed_fbank.len_feat(logspec_features_disk.shape, window_size,step_size)
        required_shape[1] = feat_gen_shape[1]
        required_shape[0] += feat_gen_shape[0]

    X_data = np.zeros(required_shape,dtype='float32')
    y_data = np.zeros(required_shape[0],dtype='uint8')

    #now we will load the npy files into memory and generate features
    pos = 0
    for myid,myclass in itertools.izip(ids,classes):
            feat = loadIdFeat(myid,X_data.dtype,window_size,step_size)
            feat_len = feat.shape[0]
            feat_dim = feat.shape[1]
            y_data[pos:pos+feat_len] = myclass
            X_data[pos:pos+feat_len] = feat 

            pos += feat_len

    return X_data,y_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate FBANK features (=logarithmic mel frequency filter banks) for all supplied files in file list (txt).')
    parser.add_argument('-f', '--filelists-dir', dest="filelists", help='file list basedir', default='corpus/voxforge/', type=str)
    parser.add_argument('-v', dest='verbose', help='verbose output', action='store_true', default=False)
    parser.add_argument('-l', '--langs', dest='langs', help='comma seperated list of lang codes',type=str,default='de,fr')
    parser.add_argument('-m', '--max', dest='max_samples', help='max utterance samples per language',type=int,default=-1)
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)
    parser.add_argument('-e', '--epochs',dest='epochs', help='number of supervised finetuning epochs', default = 20, type=int)
    parser.add_argument('-p', '--pretrain-epochs',dest='pretrainepochs', help='number of unsupervised epochs', default = 10, type=int)

    window_size = 21
    step_size = 15

    args = parser.parse_args()

    langs = (args.langs).split(',')
    lang2num = {}
    for i,lang in enumerate(langs):
        lang2num[lang] = i
    
    all_ids = []
    classes = []

    for lang in langs:
        ids = unspeech_utils.loadIdFile(args.filelists+'all_'+lang+'.txt',basedir=args.basedir)
        for myid in ids[:args.max_samples]:
            all_ids.append(myid)
            classes.append(lang2num[lang])

    y_all = np.asarray(classes)
    sss = StratifiedShuffleSplit(y_all, 1, test_size=0.1, random_state=0)
    
    train_index, test_index = iter(sss).next()

    X_ids_train, X_ids_test = [all_ids[index] for index in train_index], [all_ids[index] for index in test_index]
    y_train_flat, y_test_flat = y_all[train_index], y_all[test_index]

    #The vectors in X and corresponding classes y are now only portions of the signal of an id
    X_train,y_train = loadTrainData(X_ids_train, y_train_flat, window_size, step_size)

    #Scale mean of all training vectors
    std_scale = StandardScaler(copy=False, with_mean=True, with_std=False).fit(X_train)
    
    X_train = std_scale.transform(X_train)

    print 'Done loading data, trainsize:', len(X_ids_train)
    print 'testsize:', len(X_ids_test)

    print 'Distribution of classes in train data:'
    print itemfreq(y_train)
    print 'Distribution of classes in test data:'
    print itemfreq(y_test_flat)

    #hid_layer_units = 2560 
    hid_layer_units = 1000

    clf = DBN([X_train.shape[1],hid_layer_units, hid_layer_units, hid_layer_units, hid_layer_units, len(langs)],
            #dropouts=100,
            learn_rates=0.001,
            learn_rates_pretrain=0.00001,
            minibatch_size=200,
            #learn_rate_decays=0.9,
            epochs_pretrain=args.pretrainepochs,
            epochs=args.epochs,
            #use_re_lu=True,
            verbose=1)

    print 'fitting dbn...'
    clf.fit(X_train, y_train)

    print 'done!'

    y_pred = []
    for myid,y in itertools.izip(X_ids_test,y_test_flat):
        utterance = loadIdFeat(myid,X_train.dtype,window_size, step_size)
        utterance = std_scale.transform(utterance) 
        local_pred = clf.predict(utterance)
        voting = np.bincount(local_pred)
        #majority voting
        pred = np.argmax(voting)
        y_pred.append(pred)

    print "Accuracy:", accuracy_score(y_test_flat, y_pred)
    print "Classification report:"
    print classification_report(y_test_flat, y_pred)
    print lang2num
    print "Confusion matrix:\n%s" % confusion_matrix(y_test_flat, y_pred)
