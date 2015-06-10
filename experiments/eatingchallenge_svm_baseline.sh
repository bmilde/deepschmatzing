#!/bin/bash

THEANO_FLAGS="device=gpu2" time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_60fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -p 500 -e 500 -r 0.02 -m -1 -s models/post_submission/svm_baseline -cv -b svm -bonly -hu 500 -bhu 1000 -l cnn --use_linear_confidence -w 20 -d 0.1,0.2,0.3,0.4,0.5,0.5,0.5,0.5
