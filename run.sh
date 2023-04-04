#!/bin/bash

python main_pretrain.py --model vit1d_large --dataset fastmri --input_size 320 --patch_direction pe --batch_size 32 --epochs 100
python main_pretrain.py --model mae1d_large --dataset fastmri --input_size 320 --patch_direction pe --batch_size 32 --epochs 100
python main_pretrain.py --model vit2d_large --dataset fastmri --input_size 320 --batch_size 16 --epochs 100
python main_pretrain.py --model mae2d_large --dataset fastmri --input_size 320 --batch_size 16 --epochs 100
