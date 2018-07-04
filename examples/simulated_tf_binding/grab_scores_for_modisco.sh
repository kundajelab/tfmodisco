#!/usr/bin/env bash

FILE="scores.h5"
if [ ! -f "$FILE" ]
then
    curl -o $FILE https://raw.githubusercontent.com/AvantiShri/model_storage/23d8f3ffc89af210f6f0bf7e65585eff259ba672/modisco/scores.h5

else
    echo "File $FILE exists already"
fi

FILE="sequences.simdata.gz"
if [ ! -f "$FILE" ]
then
    wget https://raw.githubusercontent.com/AvantiShri/model_storage/db919b12f750e5844402153233249bb3d24e9e9a/deeplift/genomics/sequences.simdata.gz
else
    echo "File sequences.simdata.gz exists already"
fi
