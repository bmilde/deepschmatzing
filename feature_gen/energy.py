import numpy as np

#estimate sound intensity/energy based on logspec

def avgEnergy(energy):
    avg_energy = float(np.add.reduce(energy))/float(len(energy))
    return avg_energy

def getEnergy(spec):
    energy = np.einsum('ij,ij -> i', spec, spec)
    return energy

def filterSpec(spec,filter_divisor=3.0):
    energy = getEnergy(spec)
    avg_energy = avgEnergy(energy)
    print avg_energy
    first = np.argmax(energy > (avg_energy / filter_divisor))
    last = len(energy) - np.argmax(energy[::-1] > (avg_energy / filter_divisor))
    print first,last
    #sanity check
    if first < last and first >= 0 and last > 0:
        return spec[first:last]
    else:
        print 'filterSpec: warning, not filtering spec because first:',first,'last:',last
        return spec
