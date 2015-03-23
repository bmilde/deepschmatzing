import sys
from collections import defaultdict
import argparse
import numpy as np
import theano
import lasagne
from lasagne_addon import *

from linear_confidence_model import *

import functools

#higher recrusion limit for large models
sys.setrecursionlimit(10000)

from lasagne import layers
from lasagne import init

from sklearn import svm
from sklearn.cross_validation import KFold

from lasagne.updates import *
from nolearn.lasagne import NeuralNet,BatchIterator

from nolearn_addon import *
#these would be faster for 
#try:
#from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
#from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer

import gc

#except ImportError:

Conv2DLayer = layers.Conv2DLayer
MaxPool2DLayer = layers.MaxPool2DLayer

from lasagne.nonlinearities import rectify,softmax

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report,recall_score,accuracy_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.preprocessing import StandardScaler
from sklearn.lda import LDA
from sparse_filtering.sparse_filtering import SparseFiltering

from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

import sklearn
import itertools

from sklearn import linear_model

from feature_gen.energy import getEnergy

#plotting
from pylab import *
import matplotlib.pyplot

from utils import unspeech_utils,mean_substract
from feature_gen import energy,windowed_fbank

import scipy.stats
from scipy.stats import itemfreq

#pickle currently buggy with nolearn, using dill as replacement
#import cPickle as pickle
import dill as pickle
import os.path

def float32(k):
    return np.cast['float32'](k)

def afterEpoch(nn, train_history):
    #np.set_printoptions(threshold='nan')
    weights = [w.get_value() for w in nn.get_all_params()]
    #print weights

# Simple majority vote over many classifier class probabilities
def majority_vote(proba):
    return np.bincount(np.argmax(proba, axis=1))

def weighted_majority_vote(proba,weights):
    return np.bincount(np.argmax(proba, axis=1),weights=weights)

def serialize(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=-1)

def load(filename):
    p = None
    with open(filename, 'rb') as f:
        p = pickle.load(f)
    return(p)

# Load and construct feature vectors for a single logspec file id
def loadIdFeat(myid,dtype, window_size, step_size, stride, energy_filter=1.2):
    logspec_features = np.load(myid+'.logspec.npy')
    if(logspec_features.dtype != dtype):
        logspec_features = logspec_features.astype(dtype, copy=False)

    logspec_features_filtered = energy.filterSpec(logspec_features,energy_filter)
    feat = windowed_fbank.generate_feat(logspec_features_filtered,window_size,step_size,stride)
    return feat

def loadBaselineData(ids):
    feat_list = []
    for utterance_id in ids:
        utt_feat = np.load(utterance_id+'.baseline_feat.npy')
        feat_list += [utt_feat]

    X = np.array(feat_list)

    return X

# Load specgrams and generate windowed feature vectors
def loadTrainData(ids,classes,window_size,step_size,stride,baseline_X=None):
    
    #iterate through all files and find out the space needed to store all data in memory
    required_shape = [0,0]
    for myid in ids:
        #do not load array into memory yet
        #logspec_features_disk = np.load(myid+'.logspec.npy',mmap_mode='r')
        #feat_gen_shape = windowed_fbank.len_feat(logspec_features_disk.shape, window_size,step_size,stride)
        feat_gen_shape = loadIdFeat(myid,'float32',window_size,step_size,stride).shape
        required_shape[1] = feat_gen_shape[1]
        required_shape[0] += feat_gen_shape[0]

    # explicitly set X data to 32bit resolution and y data to 8 bit (256 possible classes)
    X_data = np.zeros(required_shape,dtype='float32')
    y_data = np.zeros(required_shape[0],dtype='uint8')

    if baseline_X != None:
        X2_data = np.zeros((required_shape[0],baseline_X.shape[1]),dtype='float32')

    #now we will load the npy files into memory and generate features
    pos = 0
    i = 0

    for myid,myclass in itertools.izip(ids,classes):
            feat = loadIdFeat(myid,'float32',window_size,step_size,stride)
            feat_len = feat.shape[0]
            feat_dim = feat.shape[1]
            y_data[pos:pos+feat_len] = myclass
            X_data[pos:pos+feat_len] = feat 

            if baseline_X != None:
                X2_data[pos:pos+feat_len] = baseline_X[i]

            pos += feat_len
            i +=1

    if baseline_X == None:
        return X_data,y_data
    else:
        return X_data,X2_data,y_data

#Model configurration hyper parameters
class ModelConfig:
    def __init__(self,learner,window_sizes,step_sizes,strides,class2num,max_epochs=1000,use_sparseFiltering=False,use_pca=True,pca_whiten=False,pca_components=100,learn_rates=0.1,momentum=0.9,minibatch_size=256,hid_layer_units=512,hid_layer_units_baseline = 512,dropouts=None,random_state=0,early_stopping_patience=100,iterations=1,computeBaseline=True,baselineClassifier = 'svm',mergeBaseline = False,use_linear_confidence = False, weightsFile=''):
        self.deep_learner = learner
        
        #feature generation and class config
        self.window_sizes = window_sizes
        self.no_classifiers = len(window_sizes)
        self.step_sizes = step_sizes
        self.strides = strides
        self.pca_components = pca_components
        self.class2num = class2num
        self.max_epochs = max_epochs
        self.epochs = max_epochs
        self._no_langs = len(class2num.keys())
        self._no_classes = self._no_langs
        self.use_pca = use_pca
        self.use_lda = False
        self.use_sparseFiltering = use_sparseFiltering
        self.pca_whiten = pca_whiten

        #cnn/dnn config
        self.learn_rates = learn_rates
        self.minibatch_size = minibatch_size
        self.hid_layer_units = hid_layer_units
        self.hid_layer_units_baseline = hid_layer_units_baseline
        self.dropouts = dropouts
        self.computeBaseline = computeBaseline
        self.baselineClassifier = baselineClassifier
        
        if dropouts == None:
            dropouts=[0.1,0.2,0.3,0.5]
        elif isinstance( dropouts, ( int, long ) ):
            dropouts_int = dropouts
            dropouts=[dropouts_int for x in xrange(4)]

        self.random_state = random_state
 
        self.early_stopping_patience = early_stopping_patience
        self.momentum = momentum

        self.iterations = 1

        self.mergeBaseline = mergeBaseline
        self.use_linear_confidence = use_linear_confidence
        self.weightsFile = weightsFile

#Model class that can be pickeled and once trained can be used for classification
class MyModel:
    def __init__(self, config):
        self._transforms = []
        self._dbns = []
        self._confidences = []
        self.baseline_clf = None
        self.baseline_transforms = []
        self.config = config
        self.results = {}

    #trains a baseline classifier on the baseline feature set
    def trainBaseline(self,ids,classes):
        def getTreeClf():
            return RandomForestClassifier(n_estimators=self.config.hid_layer_units_baseline, n_jobs=-1, random_state=42, oob_score=True, verbose=1)
        print 'Using',self.config.baselineClassifier,'as baseline classifier'
       
        if self.config.baselineClassifier.lower() == 'none':
            return

        X = loadBaselineData(ids)
        y = np.array(classes,dtype=np.int32)

        scaler = StandardScaler(copy=False, with_mean=True, with_std=True).fit(X) #mean_substract.MeanNormalize(copy=False).fit(X)
        X = scaler.transform(X)
        self.baseline_transforms.append(scaler)

        transform_clf = None

        if self.config.baselineClassifier.lower() == 'svm':
            self.baseline_clf = svm.LinearSVC(C=1.0)
            X = X.astype(np.float64)
        if self.config.baselineClassifier.lower() == 'trees':
            self.baseline_clf = getTreeClf()
        if self.config.baselineClassifier.lower() == 'trees2x':
            transform_clf = getTreeClf()
            self.baseline_clf = getTreeClf()

        if self.config.baselineClassifier.lower() == 'dnn' or self.config.baselineClassifier.lower() == 'trees_dnn':
            if self.config.baselineClassifier.lower() == 'trees_dnn':
                transform_clf = getTreeClf()
                X = transform_clf.fit(X,y).transform(X)
                self.baseline_transforms.append(transform_clf)
                transform_clf = None

            y = y.astype(np.int32)
            X = X.astype(np.float32)
            unspeech_utils.shuffle_in_unison(X,y)
            print 'classes:',self.config._no_classes,'hid layers:',self.config.hid_layer_units_baseline
            self.baseline_clf = self.stdDnn((None, X.shape[1]))

        if self.config.baselineClassifier.lower() == 'trees_svm':
            transform_clf = getTreeClf()
            self.baseline_clf = svm.LinearSVC(C=1.0)

        print 'Configured baseline clf to:',self.baseline_clf
        print 'Transform clf:',transform_clf

        if transform_clf != None:
            print 'Fitting transform clf...'
            transform_clf.fit(X,y)
            X = transform_clf.transform(X)
            self.baseline_transforms.append(transform_clf)

        if self.baseline_clf!=None:
            print 'Fitting main classifier...'
            self.baseline_clf.fit(X,y)

    def stdDnn(self,input_shape,epochs=1000):
        return NeuralNet(
                layers=[  # three layers: one hidden layer
                        ('input', layers.InputLayer),
                        ('hidden1', layers.DenseLayer),
                        ('dropout1', layers.DropoutLayer),
                        ('hidden2', layers.DenseLayer),
                        ('dropout2', layers.DropoutLayer),
                        ('hidden3', layers.DenseLayer),
                        ('output', layers.DenseLayer),
                        ],
                        # layer parameters:

                        hidden1_nonlinearity = rectify, hidden2_nonlinearity = rectify, hidden3_nonlinearity = rectify,
                        dropout1_p=0.5,
                        dropout2_p=0.5,

                        input_shape=input_shape,  # a x b input pixels per batch
                        hidden1_num_units=self.config.hid_layer_units,  # number of units in hidden layer
                        hidden2_num_units=self.config.hid_layer_units,  # number of units in hidden layer
                        hidden3_num_units=self.config.hid_layer_units,  # number of units in hidden layer

                        output_num_units=self.config._no_classes,
                        output_nonlinearity=lasagne.nonlinearities.softmax,

                        eval_size=0.0,

                        on_epoch_finished=[AdjustVariable('update_learning_rate', start=self.config.learn_rates, stop=0.001),
                                            AdjustVariable('update_momentum', start=self.config.momentum, stop=0.99),],
                                            #EarlyStopping(patience=self.config.early_stopping_patience)],
                        #batch_iterator_test=MyBatchIterator(128,forced_even=True),
                        #batch_iterator_train=MyBatchIterator(128,forced_even=True),

                        # optimization method:
                        update=nesterov_momentum,

                        update_learning_rate=theano.shared(float32(self.config.learn_rates)),
                        update_momentum=theano.shared(float32(self.config.momentum)),

                        #update
                        #update_learning_rate=self.config.learn_rates,
                        #update_momentum=momentum,

                        regression=False,  # flag to indicate we're dealing with regression problem
                        max_epochs=epochs,  # we want to train this many epochs
                        verbose=1,
                        ) 

    def energyFeature(self,X):
        X_energy = getEnergy(X)
        #addes 1 to the shape of X_energy, and makes it possible to e.g. vstack it to some other feature vector
        X_energy = np.expand_dims(X_energy, axis=1)
        return X_energy

    #Prepare corpus/generate (raw) features and train classifier
    def trainClassifier(self,all_ids,classes,window_size,step_size,stride,deep_learner):
        
        print 'Using', deep_learner, 'as classifier.'
        print 'hid_layer_units',self.config.hid_layer_units
        print 'use_sparseFiltering:',self.config.use_sparseFiltering
        print 'use_pca',self.config.use_pca
        print 'use_lda',self.config.use_lda

        y_all = np.asarray(classes)
        X_train,y_train = loadTrainData(all_ids, y_all, window_size, step_size,stride)

        #shuffle training vectors (inplace) for minibatch gradient descent optimisers
        unspeech_utils.shuffle_in_unison(X_train,y_train)

        X_energy = self.energyFeature(X_train)

        #Scale mean of all training vectors
        std_scale = mean_substract.MeanNormalize(copy=False).fit(X_train)
        X_train = std_scale.transform(X_train)

        transform_clf = None

        #Sanity check and warning, gpu needs float32
        if(X_train.dtype != 'float32'):
            print 'Warning, training data was not float32 after mean substract: ', X_train.dtype
            X_train = X_train.astype('float32', copy=False)

        #todo: switch to enable data caching / saving
        #np.save('data/X_train',X_train)
        #np.save('data/y_train',y_train)
        #np.save('data/X_train_mean',std_scale._mean)

        for iteration in xrange(self.config.iterations):
            print 'iteration ', iteration,'/',self.config.iterations
            print 'Starting dimensionally reduction'

            #todo rename pca to dim reduction
            if self.config.use_pca:
                print 'using rpca'
                pca = RandomizedPCA(n_components=self.config.pca_components, copy=False,
                               whiten=self.config.pca_whiten, random_state=random_state)
                #Pca fit
                pca.fit(X_train)
                X_train = pca.transform(X_train)
            else:
                pca = None

            if self.config.use_lda:
                print 'using LDA'
                pca = LDA(n_components=self.config.pca_components)
                pca.fit(X_train,y_train)
                #lda fit
                X_train = pca.transform(X_train)

            if self.config.use_sparseFiltering:
                print 'using sparseFilterTransform'
                #pca = sparseFilterTransform(N=hid_layer_units)
                pca = SparseFiltering(n_features=self.config.hid_layer_units, maxfun=self.config.max_epochs, iprint=1, stack_orig=False)
                pca.fit(X_train,y_train)
                print 'fitted data, now transforming...'
                print 'Shape before transform: ',X_train.shape
                X_train = pca.transform(X_train)
                print 'Shape after transform: ',X_train.shape
                print 'ytrain shape:',y_train.shape

            #if self.config.use_pca or self.config.use_lda or self.config.use_sparseFiltering:
            #    np.save('data/X_train_transformed',X_train)


            print 'Done loading and transforming data, traindata size: ', float(X_train.nbytes) / 1024.0 / 1024.0, 'MB' 

            print 'Distribution of classes in train data:'
            print itemfreq(y_train),self.config._no_langs

            if(X_train.dtype != 'float32'):
                print 'Warning, training data was not float32 after dim reduction: ', X_train.dtype
                X_train = X_train.astype('float32', copy=False)

            if deep_learner=='trees':
                print 'Using trees classifier... (2 pass)'
                
                transform_clf = None#RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42, oob_score=True, verbose=1, compute_importances=True)
                clf = RandomForestClassifier(n_estimators=self.config.hid_layer_units, n_jobs=-1, random_state=42, oob_score=True, verbose=1)

                #print 'Feature selection...'
                #print 'X_train shape:', X_train.shape
                #X_train = transform_clf.fit(X_train, y_train).transform(X_train)
                #print 'X_train after selection:', X_train.shape
                #print transform_clf.feature_importances_
                
                #clf = ExtraTreesClassifier(n_estimators=1000,
                #                           max_features='auto',
                #                           n_jobs=-1,
                #                           random_state=42)

            elif deep_learner=='svm':

                clf = svm.LinearSVC(C=1.0)

            elif deep_learner=='cnn':

                momentum = 0.9
                print 'conf: momentum:',self.config.momentum,'self.learn_rates:',self.config.learn_rates

                feat_len = X_train.shape[1] / window_size
                window_len = window_size
                
                print '2D shape window:',window_len,'x','featlen:',feat_len
                #todo: make sure y and x coordinates are not mixed up
                X_train = X_train.reshape(-1, 1, window_len, feat_len)

                print 'X_train shape:',X_train.shape 

                y_train = y_train.astype(np.int32)

                clf = NeuralNet(
                    layers=[
                        ('input', layers.InputLayer),
                        ('conv1', Conv2DLayer),
                        ('pool1', MaxPool2DLayer),
                        ('dropout1', layers.DropoutLayer),
                        ('conv2', Conv2DLayer),
                        ('pool2', MaxPool2DLayer),
                        ('dropout2', layers.DropoutLayer),
                        ('conv3', Conv2DLayer),
                        ('pool3', MaxPool2DLayer),
                        ('dropout3', layers.DropoutLayer),
                        ('hidden4', layers.DenseLayer),
                        ('dropout4', layers.DropoutLayer),
                        ('hidden5', layers.DenseLayer),
                        ('output', layers.DenseLayer),
                        ],
                    input_shape=(None, 1, window_len, feat_len),
                    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
                    dropout1_p=self.config.dropouts[0],
                    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
                    dropout2_p=self.config.dropouts[1],
                    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
                    dropout3_p=self.config.dropouts[2],
                    hidden4_num_units=self.config.hid_layer_units, hidden5_num_units=self.config.hid_layer_units,
                    dropout4_p=self.config.dropouts[3],
                    output_num_units=self.config._no_classes, 
                    
                    conv1_nonlinearity = rectify, conv2_nonlinearity = rectify, conv3_nonlinearity = rectify,
                    hidden4_nonlinearity = rectify, hidden5_nonlinearity = rectify,
                    output_nonlinearity=lasagne.nonlinearities.softmax,
                   
                    eval_size=0.01,

                    on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=self.config.learn_rates, stop=0.0001),
                        AdjustVariable('update_momentum', start=self.config.momentum, stop=0.999),
                        EarlyStopping(patience=self.config.early_stopping_patience),
                    ],

                    batch_iterator_train=ShufflingBatchIteratorMixin(batch_size=512),
                    batch_iterator_test=BatchIterator(batch_size=512),
                    
                    #update=rmsprop,
                    #update_learning_rate=1.0,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(float32(self.config.learn_rates)),
                    update_momentum=theano.shared(float32(self.config.momentum)),

                    regression=False,
                    max_epochs=self.config.max_epochs,
                    verbose=1,
                    
                    #w=ReluNormal()
                    #W=init.Uniform() 
                    )
                if self.config.weightsFile != '':
                    clf.load_weights_from(self.config.weightsFile)
            else:

                print 'Using DNN as classifier'

                y_train = y_train.astype(np.int32)
                X_train = X_train.astype(np.float32)

                print 'conf: momentum:',self.config.momentum,'self.learn_rates:',self.config.learn_rates

                print 'X_train type:', X_train.dtype
                print 'y_train type:', y_train.dtype
                
                clf = self.stdDnn((None, X_train.shape[1]),self.config.max_epochs)

            print 'fitting classifier...',deep_learner
            clf.fit(X_train, y_train)

            if deep_learner=='trees':
                print clf.feature_importances_
                serialize(clf.feature_importances_, 'models/feature_importances_iter'+str(iteration)+'.pickle')
            
            print 'done!'

            confidence_clf = None

            if self.config.mergeBaseline:
                X_train,y_train = self.clf_embedding(all_ids,y_all,(std_scale,pca,transform_clf),clf,window_size, step_size, stride)
                print 'Embedded X_train shape:',X_train
                merged_clf = self.stdDnn((None, X_train.shape[1])) 
                #shuffle training vectors (inplace) for minibatch gradient descent optimisers
                unspeech_utils.shuffle_in_unison(X_train,y_train)
                merged_clf.fit(X_train, y_train)

            if self.config.use_linear_confidence:
                #trains a linear regression model predicting confidence of
                y_train_clf_proba = clf.predict_proba(X_train)
                y_train_clf = np.argmax(y_train_clf_proba, axis=1) 
                mask = np.equal(y_train, y_train_clf)
                print 'shapes (energy/train):',X_energy.shape,y_train_clf_proba.shape
                #y_train_clf_proba = np.hstack([y_train_clf_proba,X_energy])
                print 'new shape:',y_train_clf_proba.shape
                confidence_y = mask.astype(np.float32)
                print 'confidence value distribution:',np.bincount(confidence_y.astype(np.int32))
                confidence_clf = LinearConfidenceModel(num_classproba=self.config._no_classes) #linear_model.ElasticNet(normalize=True) #linear_model.Ridge(alpha = .5)
                confidence_clf.fit([y_train_clf_proba,X_energy],confidence_y)
            
            if iteration < self.config.iterations-1:
                #restrict samples to the ones that the classifier can correctly distinguish
                y_train_clf = clf.predict(X_train)
                mask = np.equal(y_train, y_train_clf)
                y_train[mask] = self.config._no_langs
                self.config._no_classes += 1

        #hack to free now unneeded memory

        #X_train.resize((0,0),refcheck=False)
        #y_train.resize((0,0),refcheck=False)
        #X_energy.resize((0,0),refcheck=False)
        
        del X_train
        del y_train
        del X_energy
        gc.collect()

        return (std_scale,pca,transform_clf),clf,confidence_clf

    def fit(self,all_ids,classes):
        self._dbns,self._scalers,self.X_ids_train,self.X_ids_test,self.y_test_flat = [],[],None,None,None

        #print 'len ids:', len(all_ids), 'classes:', len(classes)

        if self.config.computeBaseline:
            self.trainBaseline(all_ids,classes)

        for i in xrange(self.config.no_classifiers):
            if self.config.deep_learner:
                print 'Train #',i,' dbn classifier with window_size:',self.config.window_sizes[i],'step_size=',self.config.step_sizes[i]
                transforms,clf,confidence_clf = self.trainClassifier(all_ids,classes,self.config.window_sizes[i],self.config.step_sizes[i],self.config.strides[i],self.config.deep_learner)
                self._dbns.append(clf)
                self._transforms.append(transforms)
                self._confidences.append(confidence_clf)

    #fuse frame probabiltites, defaults to geometric mean
    def fuse_op(self,frame_proba, op=scipy.stats.gmean, normalize=True):
        no_classes = frame_proba.shape[0]
        
        #nothing to fuse
        if frame_proba.shape[1] < 2:
            return frame_proba

        fused_proba = op(frame_proba)
        
        if len(fused_proba) < self.config._no_langs:
            fused_proba.resize((self.config._no_langs,))
        elif len(fused_proba) > self.config._no_langs:
            fused_proba = fused_proba[:self.config._no_langs]

        if normalize:
            return unspeech_utils.normalize_unitlength(fused_proba)
        else:
            return fused_proba

    def inspect_predict(self,utterance_id):
        clf,window_size,step_size,stride = self._dbns[0],self.config.window_sizes[0],self.config.step_sizes[0],self.config.strides[0]
        
        logspec_features = np.load(utterance_id+'.logspec.npy')
        
        utterance = loadIdFeat(utterance_id,'float32',window_size, step_size, stride)
        
        subplot(411)
        imshow(logspec_features.T, aspect='auto', interpolation='nearest')

        subplot(412)
        imshow(utterance.T, aspect='auto', interpolation='nearest')
       
        for transform in self._transforms:
            if transform != None:
                utterance = transform.transform(utterance)
         
        subplot(413)
        imshow(utterance.T, aspect='auto', interpolation='nearest')

        print 'calling classifier',utterance.dtype,utterance.shape
        frame_proba = clf.predict_proba(utterance)

        subplot(414)
        imshow(frame_proba.T, aspect='auto', interpolation='nearest')

        show()

    #use the transform function on the base classifier as base embedding (for dnn and cnn the last layer representation)
    def clf_embedding(self,utt_ids, utt_y, transforms, clf,window_size, step_size, stride):
        print 'Building classifier embedding...'
        baseline_X = self.baseline_embedding(utt_ids)
        X_train,X2_train,y_train = loadTrainData(utt_ids, utt_y, window_size, step_size, stride, baseline_X)
        
        for transform in transforms:
            if transform != None:
                utterances = transform.transform(X_train)

        if(utterances.dtype != 'float32'):
            #print 'Warning, training data was not float32: ', utterance.dtype
            utterances = utterances.astype('float32', copy=False)

        #2D reshape for cnn
        if self.config.deep_learner=='cnn':
            utterances = utterances.reshape(-1, 1, window_size, utterance.shape[1] / window_size)
        return np.hstack(clf.transform(utterances),X2_train),y_train

    def baseline_embedding(self, utt_ids):
        print 'Building baselinge embedding...'
        X = loadBaselineData(utt_ids)

        for scaler in self.baseline_transforms:
            scaler.transform(X)

        X = X.astype(np.float32)

        return self.baseline_clf.transform(X)

    #predict a whole
    def predict_utterance(self,utterance_id):
        if not self._dbns:
            return
        
        voting = []
        multi_pred = []
        for clf,confidence_clf,transforms,window_size,step_size,stride in itertools.izip(self._dbns,self._confidences,self._transforms,self.config.window_sizes,self.config.step_sizes,self.config.strides):
            utterance = loadIdFeat(utterance_id,'float32',window_size, step_size, stride)
            X_energy = self.energyFeature(utterance)
            
            for transform in transforms:
                if transform != None:
                    utterance = transform.transform(utterance)

            if(utterance.dtype != 'float32'):
                #print 'Warning, training data was not float32: ', utterance.dtype
                utterance = utterance.astype('float32', copy=False)

            #2D reshape for cnn
            if self.config.deep_learner=='cnn':
                utterance = utterance.reshape(-1, 1, window_size, utterance.shape[1] / window_size)

            #hard decision per frame, agg with majority voting
            #print 'calling classifier',utterance.dtype,utterance.shape
            frame_proba = clf.predict_proba(utterance)
            frame_proba_log = np.log(frame_proba)
            frame_proba_with_energy = np.hstack([frame_proba,X_energy])

            local_vote2 = self.fuse_op(frame_proba,op=majority_vote)

            if confidence_clf:
                weights = confidence_clf.predict([frame_proba,X_energy])
                frame_proba_weighted = frame_proba * np.array([weights]).T
                local_vote3 = self.fuse_op(frame_proba_weighted,op=np.add.reduce)
            else:
                local_vote3 = self.fuse_op(frame_proba,op=np.add.reduce)
            
            if confidence_clf:
                local_vote4 = self.fuse_op(frame_proba,op=functools.partial(weighted_majority_vote,weights=weights))
            else:
                local_vote4 = self.fuse_op(frame_proba,op=np.multiply.reduce)

            #local_vote5 = self.fuse_op(frame_proba,op=np.maximum.reduce)

            if confidence_clf and self.baseline_clf!=None:
                frame_proba_baseline = self.baselinePredictProba([utterance_id])[0]
                local_vote6 = self.fuse_op(frame_proba+frame_proba_baseline,op=functools.partial(weighted_majority_vote,weights=weights))
            else:
                local_vote6 = self.fuse_op(frame_proba_log,op=np.add.reduce)

            #local_vote3 = self.fuse_op(frame_proba,op=np.multiply.reduce)
            #local_vote4 = self.fuse_op(frame_proba,op=np.maximum.reduce)

            voting += [local_vote2,local_vote3,local_vote4,local_vote6]
            multi_pred += [np.argmax(local_vote2),np.argmax(local_vote3),np.argmax(local_vote4),np.argmax(local_vote6)]
            #multi_pred_names = ['hmean','gmean','majority_vote','add.reduce','multiply.reduce','maximum.reduce','log add.reduce']
            multi_pred_names = ['majority_vote',
                    'weighted_majority_add' if confidence_clf else 'add',
                    'weighted_majority_vote' if confidence_clf else 'mul',
                    'baseline_merge']

        #print voting

        #majority probability voting on utterance
        pred = np.argmax(np.add.reduce(voting))
        return pred,multi_pred,multi_pred_names
   

    def baselinePredictProba(self,dev_ids):
        X = loadBaselineData(dev_ids)

        for scaler in self.baseline_transforms:
            X = scaler.transform(X)

        X = X.astype(np.float32)

        return self.baseline_clf.predict_proba(X)

    def baselinePredict(self,dev_ids):
        X = loadBaselineData(dev_ids)

        for scaler in self.baseline_transforms:
            X = scaler.transform(X)

        X = X.astype(np.float32)

        return self.baseline_clf.predict(X)

    def performance_on_set(self,dev_ids,dev_classes,class2num):
        #aggregated predictions by geometric mean
        y_pred = []
        #multiple single predictions of the classifiers
        y_multi_pred = []

        print 'Performance on dev set:'

        return_recall_score = 0

        #2 class labels out of a multiclass problem, i.e. 0 is one class and 1 is everything else
        dev_classes_2class = (np.array(dev_classes)!=0).astype(np.int32)

        if self.baseline_clf:

            y_pred_baseline =self.baselinePredict(dev_ids)

            return_recall_score = recall_score(dev_classes, y_pred_baseline, average='macro', pos_label=None)
            print 'Baseline: '
            self.results['baseline'] = print_classificationreport(dev_classes, y_pred_baseline)
            print 'Baseline 2-class: '
            self.results['baseline_2c'] = print_classificationreport(dev_classes_2class, (np.array(y_pred_baseline)!=0).astype(np.int32))
        
        if len(self._dbns) > 0:
            multi_pred_names = []

            #now test on heldout ids (dev set)
            for myid in dev_ids:
                #print 'testing',myid
                pred,multi_pred,multi_pred_names = self.predict_utterance(myid)
                y_multi_pred.append(multi_pred)
                y_pred.append(pred)

            print class2num
            print 'Single classifier performance scores:'
            for i in xrange(len(multi_pred)):
                print 'Pred #',i,multi_pred_names[i],':'
                #print 'Window size', window_sizes[i],'step size',step_sizes[i]
                prediction = [pred[i] for pred in y_multi_pred]
                prediction_2class = (np.array(prediction)!=0).astype(np.int32)
                self.results[multi_pred_names[i]] = print_classificationreport(dev_classes, prediction)
                print '+'*50
                print 'Pred #',i,multi_pred_names[i],' 2-class :'
                self.results[multi_pred_names[i]+'_2c'] = print_classificationreport(dev_classes_2class, prediction_2class)
                print '*'*50
                print ' '*50
            print '*'*50
            print ' '*50
            print ' '*50
            print class2num
            print 'Fused scores:'
            self.results['fused'] = print_classificationreport(dev_classes, y_pred)
            print 'Fused scores 2-class:'
            self.results['fused_2c'] = print_classificationreport(dev_classes_2class, (np.array(y_pred)!=0).astype(np.int32))

            return_recall_score = recall_score(dev_classes, y_pred, average='macro', pos_label=None)
        return return_recall_score
        
'''generic classification report for real_classes vs. predicted_classes'''
def print_classificationreport(real_classes, predicted_classes):
    uaa= accuracy_score(real_classes, predicted_classes)
    uar= recall_score(real_classes, predicted_classes, average='macro', pos_label=None)
    report = classification_report(real_classes, predicted_classes)
    confusion = confusion_matrix(real_classes, predicted_classes)

    print 'Unweighted accuracy:', uaa
    print 'Unweighted recall:', uar 
    print 'Classification report:'
    print report
    print 'Confusion matrix:\n%s' % confusion
    return {'uaa':uaa,'uar':uar,'report':report,'confusion':confusion}

'''return the given set, with classes and name (usually train, dev, or test), as tuple of ids (list) and matching classes (list)'''
def load_set(classes,name,max_samples,class2num,withSpeakerInfo=False):
    print classes
    
    list_ids = []
    list_classes = [] 
    list_speakers = []

    for myclass in classes:
        print myclass,name
        if withSpeakerInfo:
            ids,speakers = unspeech_utils.loadIdFile(args.filelists+name+'_'+myclass+'.txt',basedir=args.basedir,withSpeakerInfo=withSpeakerInfo)
        else:
            ids = unspeech_utils.loadIdFile(args.filelists+name+'_'+myclass+'.txt',basedir=args.basedir)
        if max_samples != -1:
            ids = ids[:max_samples]
            if withSpeakerInfo:
                speakers = speakers[:max_samples]
       
        #set all speakers to the same speakers if speaker info os not available
        for myid,speaker in zip(ids,speakers if withSpeakerInfo else ['0']*len(ids)):
            list_ids.append(myid)
            list_classes.append(class2num[myclass])
            list_speakers.append(speaker)

    return list_ids,list_classes,list_speakers

def train_dev_split(all_ids, classes, speakers, dev_speaker_sel):
    train_ids, train_classes, train_speakers, dev_ids, dev_classes, dev_speakers = [],[],[],[],[],[]

    for myid,myclass,speaker in zip(all_ids, classes, speakers):
        print myid,myclass,speaker
        if speaker in dev_speaker_sel:
            dev_ids += [myid]
            dev_classes += [myclass]
            dev_speakers += [speaker]
        else:
            train_ids += [myid]
            train_classes += [myclass]
            train_speakers += [speaker]
    return train_ids, train_classes, train_speakers, dev_ids, dev_classes, dev_speakers


def createModel(args,dataset_classes,class2num):
    window_sizes = [int(x) for x in args.window_size.split(',')]
    no_classifiers = len(window_sizes)
    step_sizes = [2] 
    strides = [1]

    #dataset_classes = (args.classes).split(',')
    #class2num = {}
    #for i,myclass in enumerate(dataset_classes):
    #    class2num[myclass] = i
    
    no_classes = len(dataset_classes)
    print 'classes:',dataset_classes

    all_ids, classes, speakers = load_set(dataset_classes,'train',args.max_samples,class2num, args.test_speakers != '')
  
    print 'classes:',set(classes)
    print 'first view data elements:',zip(all_ids, classes, speakers)[:10]

    #print len(all_ids),len(classes),len(speakers)

    if args.test_speakers != '':
        dev_speaker_sel = (args.test_speakers).split(',')
        train_ids, train_classes, train_speakers, dev_ids, dev_classes, dev_speakers = train_dev_split(all_ids, classes, speakers, dev_speaker_sel)
        if len(dev_speakers) == 0:
            print 'Speakers:',args.test_speakers,'not found'
            print 'Choose one or more of:',set(train_speakers)
            sys.exit(-1)
        print 'Train speakers:',set(train_speakers),'dev speakers:',set(dev_speakers)
        print 'Train classes:',set(train_classes),'dev classes:',set(dev_classes)
    else:
        train_ids, train_classes, train_speakers = all_ids, classes, speakers
        dev_ids, dev_classes, dev_speakers = load_set(dataset_classes,'dev',args.max_samples,class2num, False)

    args_dropouts = [float(x) for x in args.dropouts.split(',')]

    if args.baseline_only:
        args.deep_learner = ''

    if args.load_model != '':
        model = load(args.load_model) 
    else:
        config = ModelConfig(args.deep_learner,
                            window_sizes,
                            step_sizes,
                            strides,
                            class2num,
                            max_epochs=args.max_epochs,
                            use_sparseFiltering=args.use_sparse,
                            use_pca=args.use_pca,
                            pca_whiten=args.pca_whiten,
                            pca_components=100,
                            learn_rates=args.learnrate,
                            momentum=args.momentum,
                            minibatch_size=256,
                            hid_layer_units=args.hiddenlayerunits,
                            hid_layer_units_baseline= args.hiddenlayerunits if args.hiddenlayerunits_baseline==0 else args.hiddenlayerunits_baseline,
                            dropouts=args_dropouts,
                            random_state=0,
                            early_stopping_patience=args.early_stopping_patience,
                            iterations=1,
                            computeBaseline=(args.baseline != ''),
                            baselineClassifier = args.baseline,
                            mergeBaseline = args.merge_baseline,
                            use_linear_confidence = args.use_linear_confidence,
                            weightsFile = args.weights_file) 

        model = MyModel(config)
        model.fit(train_ids,train_classes)
    
    model.performance_on_set(dev_ids,dev_classes,class2num)

    if (args.modelfilename != ''):
        
        print 'serialzing model...'

        if args.deep_learner == 'cnn' or args.deep_learner == 'dnn':
            print 'saving network weights'
            model._dbns[0].save_weights_to(args.modelfilename+'.weights.npy')
       
        print 'saving model...'
        serialize(model,args.modelfilename+'.pickle')
        print 'extra copy for results'
        serialize(model.results,args.modelfilename+'.results.pickle')
    return model

def crossvalidated_result_report(results):
    assert(len(results) >= 1)

    outputstr = ''
    outputstr += '*'*50 + '\n'
    combined_uar = defaultdict(list)
    combined_confusion = {}

    for result in results:
        for key in result.keys():
            if key not in combined_confusion:
                combined_confusion[key] = result[key]['confusion']
            else:
                combined_confusion[key] += result[key]['confusion']
            combined_uar[key] += [result[key]['uar']]
    
    for key in results[0].keys():
        outputstr += '+'*50 + '\n' +  ' '*50 + '\n'
        outputstr += 'CV for: ' + str(key) + '\n'
        outputstr += str(combined_confusion[key]) + '\n\n'
        uar = np.asarray(combined_uar[key])
        outputstr += 'Mean: ' + str(np.mean(uar)) + '\n'
        outputstr += 'Individual: ' +  str(uar) + '\n'
        outputstr += 'Max+: ' + str(np.amax(uar)-np.mean(uar)) + '\n'
        outputstr += 'Max-: ' + str(np.mean(uar)-np.amin(uar)) + '\n'
    print outputstr
    return outputstr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate FBANK features (=logarithmic mel frequency filter banks) for all supplied files in file list (txt).')
    parser.add_argument('-f', '--filelists-dir', dest="filelists", help='file list basedir', default='corpus/voxforge/', type=str)
    parser.add_argument('-v', dest='verbose', help='verbose output', action='store_true', default=False)
    parser.add_argument('-c', '--classes', dest='classes', help='comma seperated list of classes',type=str,default='de,fr')
    parser.add_argument('-m', '--max-samples', dest='max_samples', help='max utterance samples per language',type=int,default=-1)
    parser.add_argument('-bdir', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)
    parser.add_argument('-b', '--baseline',dest='baseline', help='compute baseline with this classifier (svm,randomforest,dnn)', default = 'svm', type=str)
    parser.add_argument('-bonly', '--baseline-only',dest='baseline_only', help='compute only baseline, specify classifier with -b', action='store_true', default = False)
    parser.add_argument('-lc','--use_linear_confidence',dest='use_linear_confidence',help='use linear confidence', action='store_true', default = False)
    parser.add_argument('-e', '--max-epochs',dest='max_epochs', help='maximum number of supervised finetuning epochs', default = 1000, type=int)
    parser.add_argument('-s', '--save-model',dest='modelfilename', help='store trained model to this filename, if specified', default = '', type=str)
    parser.add_argument('-d', '--dropouts',dest='dropouts', help='dropout param in cnn', default = '0.1,0.2,0.3,0.5', type=str)
    parser.add_argument('-hu', '--hidden-layer-units',dest='hiddenlayerunits', help='number of fully dense hidden layer units', default = 512, type=int)
    parser.add_argument('-bhu', '--hidden-layer-units-baseline',dest='hiddenlayerunits_baseline', help='number of fully dense hidden layer units for baseline (default to same as -hu if set to zero)', default = 0, type=int)
    parser.add_argument('-r', '--learnrate',dest='learnrate', help='learning rate', default = 0.01, type=float)
    parser.add_argument('-mo', '--momentum',dest='momentum', help='momentum', default = 0.9, type=float)
    parser.add_argument('-l', '--deep-learner',dest='deep_learner', help='learning rate', default = 'nolearn', type=str)
    parser.add_argument('-g', '--gpu-id',dest='gpu_id', help='GPU board to use (defaults to 0)', default = 0, type=int)
    parser.add_argument('-p', '--early-stopping-patience',dest='early_stopping_patience', help='wait this many epochs for a better result, otherwise stop training', default = 100, type=int)
    parser.add_argument('-cv', '--cross-validation', dest='crossvalidation' , help='5x cross-validated run, takes precedence over test speakers. Requires speaker information.', action='store_true', default=False)
    parser.add_argument('-t', '--test-speakers',dest='test_speakers', help='list of test speakers instead of a development or test set', default='',type=str)
    parser.add_argument('-w', '--window-size',dest='window_size', help='single window size or list of window sizes', default='11',type=str)
    parser.add_argument('--with-sparsefiltering', dest='use_sparse', help='Use sparse filtering features', action='store_true', default=False)
    parser.add_argument('--pca', dest='use_pca', help='pca reduction of feature space', action='store_true', default=False)
    parser.add_argument('--pca-whiten', dest='pca_whiten', help='pca whiten (decorellate) features', action='store_true', default=False)
    parser.add_argument('--merge-baseline', dest='merge_baseline', help='merge with baseline features (only with dnn baseline)', action='store_true', default=False)
    parser.add_argument('--preload-weights-from', dest='weights_file', help='Preload weights file', type=str, default='')
    parser.add_argument('--load-model-from', dest='load_model', help='Load model from this file, evaluate performance only', type=str, default='')

    args = parser.parse_args()

    models = []

    dataset_classes = (args.classes).split(',')
    class2num = {}
    for i,myclass in enumerate(dataset_classes):
        class2num[myclass] = i

    #do not overwrite existing files
    while(os.path.isfile(args.modelfilename+'.pickle') or os.path.isfile(args.modelfilename+'_0.pickle')):
        print 'Warning',args.modelfilename+'.pickle','exists.'
        args.modelfilename += '.new'
        print 'Choosen:',args.modelfilename,'as new filename!'
    
    if args.crossvalidation:
        modelfilename = args.modelfilename

        all_ids, classes, speakers = load_set(dataset_classes,'train',args.max_samples,class2num, True)
        speakers = list(set(speakers))
        
        kf = KFold(len(speakers), n_folds=2)

        for fold,(train_sel, dev_sel) in enumerate(kf):
            args.test_speakers = ','.join([elem for i,elem in enumerate(speakers) if i in dev_sel])
            print 'CV Fold: ', fold, 'test speakers:',args.test_speakers
            
            if modelfilename != '':
                args.modelfilename = modelfilename + ('_'+str(fold))
            
            model = createModel(args, dataset_classes, class2num)
            models.append(model)

        results = [model.results for model in models]

        txt_report = crossvalidated_result_report(results)
        with open(args.modelfilename+'.cvresults.txt','w') as report_out:
            report_out.write(txt_report)
    else:    
        createModel(args, dataset_classes, class2num)
