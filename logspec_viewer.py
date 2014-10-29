from pylab import *
from utils import unspeech_utils
import numpy as np
import argparse
from feature_gen import energy

def generatePlot(ids,num):
    myid = ids[num]
    signal, framerate = unspeech_utils.getSignal(ids[num]+'.wav')
    
    logpsec_features = np.load(ids[num]+'.logspec.npy')
    logpsec_features_filtered = energy.filterSpec(logpsec_features)
    
    subplot(311)
    imshow(logpsec_features.T, aspect='auto', interpolation='nearest')
    #specgram(signal, Fs=framerate, scale_by_freq=True, sides='default')
    subplot(312)
    plot(energy.getEnergy(logpsec_features))
    subplot(313)
    imshow(logpsec_features_filtered.T, aspect='auto', interpolation='nearest')
    show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot wav spectogram and logspec features')

    parser.add_argument('-f', '--filelist', dest="filelist", help='Process this file list, txt format. Omit filetype.', default='test_ids/nachrichten_1h', type=str)
    parser.add_argument('-i', '--inspect-id', dest='inspect_id', help='inspect this id instead of computing units', default=-1, type=int)
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)

    args = parser.parse_args()
    ids = unspeech_utils.loadIdFile(args.filelist,args.basedir)

    generatePlot(ids,args.inspect_id)
