#!/usr/bin/env bash

FILE="talgata_task0_positives_multipliers.npy"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/raw/bbd32930352e94b22b9381c28f7a55e209c98ad0/modisco/talgata_task0_positives_multipliers.npy
else
    echo "File talgata_task0_positives_multipliers.npy exists already"
fi
