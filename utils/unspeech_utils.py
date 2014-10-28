import wave
import numpy as np
import os

def readWordPosFile(filename,pos1=0,pos2=1):
    unalign_list = []
    with open(filename) as f:
        for line in f.readlines():
            split = line[:-1].split(" ")
            unalign_list.append((float(split[pos1]), float(split[pos2])))
    return unalign_list

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def loadIdFile(idfile,use_no_files=-1):
    ids = []
    with open(idfile) as f:
        ids = f.read().split('\n')[:use_no_files]
    #check if ids exist
    #ids = [myid for myid in ids if os.path.ispath(myid)]
    return ids

def getSignal(utterance):
    spf = wave.open(utterance, 'r')
    sound_info = spf.readframes(-1)
    signal = np.fromstring(sound_info, 'Int16')
    return signal, spf.getframerate()