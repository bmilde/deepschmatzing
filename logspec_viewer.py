from pylab import *
from utils import unspeech_utils
import numpy as np
import argparse
from feature_gen import energy,windowed_fbank
#from feature_gen import windowed_fbank

def generatePlot(ids,num):
    myid = ids[num]
    signal, framerate = unspeech_utils.getSignal(ids[num]+'.wav')
    
    logspec_features = np.load(ids[num]+'.logspec.npy')
    logspec_features_filtered = energy.filterSpec(logspec_features,1.2)
    
    subplot(411)
    imshow(logspec_features.T, aspect='auto', interpolation='nearest')
    #specgram(signal, Fs=framerate, scale_by_freq=True, sides='default')
    subplot(412)
    plot(energy.getEnergy(logspec_features))
    subplot(413)
    imshow(logspec_features_filtered.T, aspect='auto', interpolation='nearest')
    subplot(414)
    feat = windowed_fbank.generate_feat(logspec_features_filtered,window_size=9,step_size=3,stride=3)
    imshow(feat.T, aspect='auto', interpolation='nearest')
    show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot wav spectogram and logspec features')

    parser.add_argument('-f', '--filelist', dest="filelist", help='Process this file list, txt format. Omit filetype.', default='test_ids/nachrichten_1h', type=str)
    parser.add_argument('-i', '--inspect-id', dest='inspect_id', help='inspect this id instead of computing units', default=-1, type=int)
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)

    args = parser.parse_args()
    ids = unspeech_utils.loadIdFile(args.filelist,args.basedir)

    generatePlot(ids,args.inspect_id)
