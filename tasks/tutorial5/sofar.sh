#!/bin/bash
PID=$1
OUTFILE=$2

cat job.sh.o$PID | grep 'Trial:' > $OUTFILE
sed -i 's/Accuracy://g' $OUTFILE
sed -i 's/Trial://g' $OUTFILE
sed -i '1s/^/n,accuracy\n/' $OUTFILE
sed -i 's/[[:space:]]//g' $OUTFILE
