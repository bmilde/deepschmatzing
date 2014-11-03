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

import scipy.stats
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

    # explicitly set X data to 32bit resolution and y data to 8 bit (256 possible languages / classes)
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

class MyModel:
    def __init__(self, window_sizes,step_sizes,lang2num,pretrainepochs,epochs,learn_rates=0.001,learn_rates_pretrain=0.00001,minibatch_size=200,hid_layer_units=1000,random_state=0):
        self._scalers = []
        self._dbns = []
        #self.all_ids = all_ids
        #self.classes = classes
        self.window_sizes = window_sizes
        self.step_sizes = step_sizes
        self.lang2num = lang2num
        self.pretrainepochs = pretrainepochs
        self.epochs = epochs
        self._no_langs = len(langs)
        self.learn_rates = learn_rates
        self.learn_rates_pretrain = learn_rates_pretrain
        self.minibatch_size = minibatch_size
        self.hid_layer_units = hid_layer_units
        self.random_state = random_state

    #Prepare coprpus and train dbn classifier
    def trainClassifier(self,all_ids,classes,window_size,step_size,pretrainepochs,epochs,learn_rates=0.001,learn_rates_pretrain=0.00001,minibatch_size=200,hid_layer_units=1000,random_state=0):
        y_all = np.asarray(classes)
        sss = StratifiedShuffleSplit(y_all, 1, test_size=0.1, random_state=random_state)
        
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


        clf = DBN([X_train.shape[1],hid_layer_units, hid_layer_units, hid_layer_units, len(langs)],
                #dropouts=100,
                learn_rates=learn_rates,
                learn_rates_pretrain=learn_rates_pretrain,
                minibatch_size=minibatch_size,
                #learn_rate_decays=0.9,
                epochs_pretrain=pretrainepochs,
                epochs=args.epochs,
                #use_re_lu=True,
                verbose=1)

        print 'fitting dbn...'
        clf.fit(X_train, y_train)

        print 'done!'

        return std_scale,clf,X_ids_train,X_ids_test,y_test_flat

    def fit(self,all_ids,classes):
        self._dbns,self._scalers,self.X_ids_train,self.X_ids_test,self.y_test_flat = [],[],None,None,None

        for i in xrange(no_classifiers):
            print 'Train #',i,' dbn classifier with window_size:',window_sizes[i],'step_size=',step_sizes[i]
            std_scale,clf,self.X_ids_train,self.X_ids_test,self.y_test_flat = self.trainClassifier(all_ids,classes,self.window_sizes[i],self.step_sizes[i],self.pretrainepochs,self.epochs,hid_layer_units=self.hid_layer_units,random_state=self.random_state)
            self._dbns.append(clf)
            self._scalers.append(std_scale)
    
    def predict_utterance(self,utterance_id):
        voting = []
        multi_pred = []
        for clf,std_scale,window_size,step_size in itertools.izip(self._dbns,self._scalers,self.window_sizes,self.step_sizes):
            utterance = loadIdFeat(myid,'float32',window_size, step_size)
            utterance = std_scale.transform(utterance)
            local_pred_all = clf.predict(utterance)
            local_vote = np.bincount(local_pred_all)
            local_pred = np.argmax(local_vote)
            if len(local_vote) < no_langs:
                local_vote.resize((no_langs,))
            voting += [unspeech_utils.normalize_unitlength(local_vote)]
            multi_pred += [local_pred]
        
        #geometric mean of classifiers, majority voting on utterance
        pred = np.argmax(scipy.stats.gmean(voting))
        return pred,multi_pred 

def print_classificationreport(real_classes, predicted_classes):
    print "Accuracy:", accuracy_score(real_classes, predicted_classes)
    print "Classification report:"
    print classification_report(real_classes, predicted_classes)
    print "Confusion matrix:\n%s" % confusion_matrix(real_classes, predicted_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate FBANK features (=logarithmic mel frequency filter banks) for all supplied files in file list (txt).')
    parser.add_argument('-f', '--filelists-dir', dest="filelists", help='file list basedir', default='corpus/voxforge/', type=str)
    parser.add_argument('-v', dest='verbose', help='verbose output', action='store_true', default=False)
    parser.add_argument('-l', '--langs', dest='langs', help='comma seperated list of lang codes',type=str,default='de,fr')
    parser.add_argument('-m', '--max', dest='max_samples', help='max utterance samples per language',type=int,default=-1)
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)
    parser.add_argument('-e', '--epochs',dest='epochs', help='number of supervised finetuning epochs', default = 20, type=int)
    parser.add_argument('-p', '--pretrain-epochs',dest='pretrainepochs', help='number of unsupervised epochs', default = 10, type=int)

    no_classifiers = 2
    window_sizes = [15,26]#[11,21]
    step_sizes = [10,17]#[7,15]

    hid_layer_units = 1000

    args = parser.parse_args()

    langs = (args.langs).split(',')
    lang2num = {}
    for i,lang in enumerate(langs):
        lang2num[lang] = i
    
    no_langs = len(langs)

    all_ids = []
    classes = []

    for lang in langs:
        ids = unspeech_utils.loadIdFile(args.filelists+'all_'+lang+'.txt',basedir=args.basedir)
        for myid in ids[:args.max_samples]:
            all_ids.append(myid)
            classes.append(lang2num[lang])

    model = MyModel(window_sizes,step_sizes,lang2num,pretrainepochs=args.pretrainepochs,epochs=args.epochs)
    model.fit(all_ids,classes)

    #aggregated predictions by geometric mean
    y_pred = []
    #multiple single predictions of the classifiers
    y_multi_pred = []

    #now test on heldout ids (dev set)
    for myid in model.X_ids_test:
        pred,multi_pred = model.predict_utterance(myid)
        y_multi_pred.append(multi_pred)
        y_pred.append(pred)

    print lang2num
    print 'Single classifier performance scores:'
    for i in xrange(no_classifiers):
        print 'Clf #',i,':'
        print 'Window size', window_sizes[i],'step size',step_sizes[i]
        print_classificationreport(model.y_test_flat, [pred[i] for pred in y_multi_pred])
    print '+'*50
    print lang2num
    print 'Fused scores:'
    print_classificationreport(model.y_test_flat, y_pred)
