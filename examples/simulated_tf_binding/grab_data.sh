#!/usr/bin/env bash

FILE="talgata_task0_positives_multipliers.hdf5"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/raw/master/modisco/talgata_task0_positives_multipliers.hdf5
else
    echo "File talgata_task0_positives_multipliers.hdf5 exists already"
fi

FILE="talgata_task0_positives_scores.hdf5"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/raw/master/modisco/talgata_task0_positives_scores.hdf5
else
    echo "File talgata_task0_positives_scores.hdf5 exists already"
fi
