#!/usr/bin/env bash

FILE="task0importancescores.npy"
if [ ! -f "$FILE" ]
then
    curl -o $FILE https://raw.githubusercontent.com/AvantiShri/model_storage/adcb8ea43a964bcd6ade1e66d1e9db3605fce064/modisco/task0importancescores.npy

else
    echo "File $FILE exists already"
fi

FILE="task0hypimpscores.npy"
if [ ! -f "$FILE" ]
then
    curl -o $FILE https://raw.githubusercontent.com/AvantiShri/model_storage/adcb8ea43a964bcd6ade1e66d1e9db3605fce064/modisco/task0hypimpscores.npy
else
    echo "File $FILE exists already"
fi
FILE="sequences.txt"
if [ ! -f "$FILE" ]
then
    curl -o $FILE https://raw.githubusercontent.com/AvantiShri/model_storage/7823666c6d82bee90e03a1118a055d9af62f7894/modisco/sequences.txt
else
    echo "File $FILE exists already"
fi
