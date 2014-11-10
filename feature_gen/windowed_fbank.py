import numpy as np

#precompute size of new features
def len_feat(shape,window_size,step_size=1,stride=1):
    spec_len = shape[0]
    spec_dim = shape[1]

    if stride == 1:
        window_size_elem = window_size
    else:
        window_size_elem = int(window_size) / int(stride)
        if(int(window_size) % int(stride) != 0):
            window_size_elem += 1

    feature_len = (spec_len - window_size)/step_size
    feature_dim = spec_dim*window_size_elem
    return (feature_len,feature_dim)

#generate windowed sliced of the spectogram
def generate_feat(spec,window_size,step_size=1,stride=1):
    spec_len = spec.shape[0]
    spec_dim = spec.shape[1]
    
    feat = np.zeros(len_feat(spec.shape,window_size,step_size,stride))

    for i,spec_pos in enumerate(xrange(0, spec_len - window_size,step_size)):
        cur_slice = spec[spec_pos:spec_pos + window_size:stride]
        cur_slice = cur_slice.flatten()
        feat[i:i+1] = cur_slice
    return feat
