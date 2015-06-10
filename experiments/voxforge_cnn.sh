#!/bin/bash
time python sequence_cnn.py -bdir / -f corpus/voxforge/ -c de,fr,es,en,nl,ru,it -p 500 -e 500 -r 0.02 -m 7000 -s models/post_submission/voxforge_w18_fbank60 -b none -hu 500 -l cnn --use_linear_confidence -w 18 -d 0.1,0.2,0.3,0.4,0.5,0.5,0.5,0.5
