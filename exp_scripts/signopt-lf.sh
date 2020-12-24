#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
seed=${seed:-1}
gpu=${gpu:-"auto"}

dataset=${dataset:-CIFAR10}
model=${model:-cifar10}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset
echo 'gpu:' $gpu

cd ../
python test_attack.py \
    --attack Sign_OPT_lf \
    --dataset CIFAR10 \
    --model $model \
    --epsilon 0.031 \
    --test_batch_size 1 \
    --test_batch 100 \
    --query 40000 \
    --seed $seed \
    --save $id \
    --gpu $gpu \