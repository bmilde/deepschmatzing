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

def loadIdFile(idfile,basedir='./',use_no_files=-1,withSpeakerInfo=False):
    ids = []
    with open(idfile) as f:
        if withSpeakerInfo:
            lines = f.read().splitlines()
            ids = [line.split(' ')[0] for line in lines]
            speakers = [line.split(' ')[1] for line in lines]
        else:
            ids = f.read().splitlines()

    if use_no_files != -1:
        ids = ids[:use_no_files]
        if withSpeakerInfo:
            speakers = speakers[:use_no_files]

    #check if ids exist
    #ids = [myid for myid in ids if os.path.ispath(myid)]

    #prefix basedir
    ids = [basedir + myid for myid in ids]

    #remove .wav extension if it exists
    ids = [(myid[:-4] if myid.endswith('.wav') else myid) for myid in ids]
    
    if withSpeakerInfo:
        return ids,speakers
    else:
        return ids

def getSignal(utterance):
    spf = wave.open(utterance, 'r')
    sound_info = spf.readframes(-1)
    signal = np.fromstring(sound_info, 'Int16')
    return signal, spf.getframerate()

def normalize_unitlength(unit):
    length = np.linalg.norm(unit)
    if length != 0:
        return unit/np.linalg.norm(unit)
    else:
        return unit

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

