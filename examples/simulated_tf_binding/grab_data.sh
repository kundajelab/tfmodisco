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
    curl -o $FILE https://raw.githubusercontent.com/AvantiShri/model_storage/6a8e8d9e9e6338ca17f3d1a7cb4f54494f4e4ed0/modisco/sequences.txt
else
    echo "File $FILE exists already"
fi
