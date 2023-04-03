#!/bin/bash

python main_pretrain.py --model vit1d_large --dataset fastmri --input_size 320 --patch_direction pe --batch_size 32

python main_pretrain.py --model vit2d_large --dataset fastmri --input_size 320 --batch_size 16