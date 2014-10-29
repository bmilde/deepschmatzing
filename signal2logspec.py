import argparse
import pickle
import multiprocessing
import numpy as np
import functools
import itertools

#local imports:
from feature_gen import mfcc
#from reddit source code, nicely format progress bar
from utils import progress
from utils import unspeech_utils

#This script uses mfcc.py, from Sphinx, to compute FBank features.

#master thesis defaults: nfilt=40,frate = 200, wlen=0.025
#sphinx defaults: nfilt=40, ncep=13,lowerf=133.3333, upperf=6855.4976, alpha=0.97,samprate=16000, frate=100, wlen=0.0256, nfft=512 

def genLogspecForId(myid,nofilts,framerate,windowlen):
    signal, sampling_rate = unspeech_utils.getSignal(myid+'.wav')
    extractor = mfcc.MFCC(nfilt=nofilts,frate = framerate, wlen=windowlen, samprate=sampling_rate)
    feat = extractor.sig2logspec(signal)
    np.save(myid+'.logspec', feat)

    with open(myid+'.framepos','w') as f:
        pickle.dump(extractor.framepos,f)

    return myid+'.logspec'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate FBANK features (=logarithmic mel frequency filter banks) for all supplied files in file list (txt).')
    parser.add_argument('-f', '--filelist', dest="filelist", help='process this file list, txt format, relative filenames (see also basedir)', default='split_ids.txt', type=str)
    parser.add_argument('-v', dest='verbose', help='verbose output', action='store_true', default=False)
    parser.add_argument('-p', '--parallel', dest='use_parallel', help='parallel computing',action='store_true')
    parser.add_argument('-j', '--numjobs', dest='numjobs', help='If parallel computing is enabled, the number of jobs that run in parallel. (int, default: -1, automatic)', default=-1, type=int)
    parser.add_argument('-r', '--frate',dest='frate', help='number of frames per minute (int, default:100)', default = 100, type=int)
    parser.add_argument('-n', '--nfilt',dest='nfilt', help='number of filter banks (int, default:40)', default = 40, type=int)
    parser.add_argument('-w', '--window',dest='wlen', help='size of window, in seconds. (float, default:0.0256)', default = 0.0256, type=float)    
    parser.add_argument('-b', '--basedir',dest='basedir', help='base dir of all files, should end with /', default = './', type=str)

    args = parser.parse_args()
    ids = unspeech_utils.loadIdFile(args.filelist,basedir=args.basedir)
    genLogspecForId_partial = functools.partial(genLogspecForId,nofilts=args.nfilt,framerate=args.frate,windowlen=args.wlen)

    print 'Going to process ', len(ids), ' files.'

    if args.use_parallel:
        pool = multiprocessing.Pool()
        iterator = pool.imap(genLogspecForId_partial, ids)
    else:
        iterator = itertools.imap(genLogspecForId_partial, ids)
    
    for x in progress.prog_iterate(iterator, estimate=len(ids), verbosity=1):
        pass
