#!/bin/bash
PID=$1
OUTFILE=$2

cat job.sh.o$PID | grep 'Trial:' > $OUTFILE
sed -i 's/FstTrain://g' $OUTFILE
sed -i 's/SndTrain://g' $OUTFILE
sed -i 's/Compressa://g' $OUTFILE
sed -i 's/Trial://g' $OUTFILE
sed -i '1s/^/n,fst_train,compress,snd_train\n/' $OUTFILE
sed -i 's/[[:space:]]//g' $OUTFILE
