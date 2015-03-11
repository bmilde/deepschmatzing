append_res () {
    echo Calling $@
    echo "###############################################" >> results.txt
    echo Calling $@ >> results.txt
    time $@ >> results.txt
    echo "###############################################" >> results.txt
}


#baseline random forests

#mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -d 0.5,0.5 -p 100 -e 1000 -r 0.01 -l cnn -m -1 -s models/baseline_random_forests.pickle -t Prob01,Prob04,Prob14,Prob25 -b trees -bonly -hu 1000"
#append_res $mycmd

#baseline dnn

#mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -d 0.5,0.5 -p 100 -e 1000 -r 0.01 -l cnn -m -1 -s models/baseline_dnn.pickle -t Prob01,Prob04,Prob14,Prob25 -b dnn -bonly -hu 1000"
#append_res $mycmd

#baseline svm

#mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -d 0.5,0.5 -p 100 -e 1000 -r 0.01 -l cnn -m -1 -s models/baseline_svm.pickle -t Prob01,Prob04,Prob14,Prob25 -b svm -bonly -hu 1000"
#append_res $mycmd

#dnn

#mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -p 100 -e 1000 -r 0.01 -m -1 -s models/dnn_500_hidden.pickle -t Prob01,Prob04,Prob14,Prob25 -b none -hu 500 -l dnn -d 0.5,0.5 --use_linear_confidence -w 5"
#append_res $mycmd

#dnn

mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -p 100 -e 1000 -r 0.01 -m -1 -s models/dnn_1000_hidden.pickle -t Prob01,Prob04,Prob14,Prob25 -b none -hu 1000 -l dnn -d 0.5,0.5 --use_linear_confidence -w 5"
append_res $mycmd

#cnn

mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -p 100 -e 1000 -r 0.01 -m -1 -s models/cnn_500_hidden.pickle -t Prob01,Prob04,Prob14,Prob25 -b none -hu 500 -l cnn --use_linear_confidence"
append_res $mycmd

#cnn

mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -p 100 -e 1000 -r 0.01 -m -1 -s models/cnn_1000_hidden.pickle -t Prob01,Prob04,Prob14,Prob25 -b none -hu 1000 -l cnn --use_linear_confidence"
append_res $mycmd

#trees only
mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -p 100 -e 1000 -r 0.01 -m -1 -s models/trees_only.pickle -t Prob01,Prob04,Prob14,Prob25 -b nond -hu 1000 -l trees --use_linear_confidence"
append_res $mycmd

#trees sparse filtering
mycmd="time python sequence_cnn.py -bdir /srv/data/speech/ComParE2015/eating_40fbank_8khz/wav/ -f corpus/eating_challenge_nodev/ -c No_Food,Banana,Crisp,Nectarine,Haribo,Apple,Biscuit -p 100 -e 1000 -r 0.01 -m -1 -s models/trees_on_sparse_filtering.pickle -t Prob01,Prob04,Prob14,Prob25 -b nond -hu 1000 -l trees --with-sparsefiltering --use_linear_confidence"
append_res $mycmd
