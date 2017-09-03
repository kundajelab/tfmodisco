#!/usr/bin/env bash

FILE="talgata_task0_positives_multipliers.npy"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/blob/006ca533757587031766fe8685df1a3c7eb82b1b/modisco/talgata_task0_positives_multipliers.npy
else
    echo "File talgata_task0_positives_multipliers.npy exists already"
fi

FILE="talgata_task0_positives_scores.npy"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/blob/006ca533757587031766fe8685df1a3c7eb82b1b/modisco/talgata_task0_positives_scores.npy
else
    echo "File talgata_task0_positives_scores.npy exists already"
fi
