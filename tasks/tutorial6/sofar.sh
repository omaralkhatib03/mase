#!/bin/bash
INFILE=$1
OUTFILE=$2

cat $INFILE | grep 'Trial:' > $OUTFILE
sed -i 's/Accuracy://g' $OUTFILE
sed -i 's/Trial://g' $OUTFILE
sed -i '1s/^/n,accuracy\n/' $OUTFILE
sed -i 's/[[:space:]]//g' $OUTFILE
