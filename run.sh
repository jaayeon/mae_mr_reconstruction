#!/bin/bash
python main_pretrain.py --model vit1d_small --dataset fastmri --patch_direction pe --input_size 320 --batch_size 16 --epochs 100
python main_pretrain.py --model mae1d_small --dataset fastmri --patch_direction pe --input_size 320 --batch_size 16 --epochs 100
