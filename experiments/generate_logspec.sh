#!/bin/bash
time python signal2logspec.py -bdir /srv/data/speech/ComParE2015/eating_60fbank_8khz_aug/wav/ -n 60 -f corpus/eating_challenge/all.txt -var "_lower;_higher" -p
