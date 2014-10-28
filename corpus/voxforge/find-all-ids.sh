#!/bin/bash
run=0
#iterate over all dirs in voxforge folder
for I in `find /srv/data/speech/voxforge/ -maxdepth 1 -type d`
do
	echo $I
	L="${I: -2}"
	#exclude first row of find (usually the basedir)
	if [ $run -ne 0 ]; then
		echo $L
		find /srv/data/speech/voxforge/$L/ -name "*.wav" -printf "voxforge/$L/%P\n" > all_$L.txt
	fi
	run=$((run+1))
done
#find /srv/data/speech/voxforge/pt/ -name "*.wav" -printf "voxforge/pt/%P\n" > pt-all.txt
