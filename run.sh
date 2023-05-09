#!/bin/bash
python main_pretrain.py --model vit2d_small --dataset fastmri --input_size 320 --batch_size 16 --epochs 200 --resume 0404_vit2d_small_X4/checkpoint-best.pth
python main_pretrain.py --model mae2d_small --dataset fastmri --input_size 320 --batch_size 16 --epochs 200 --resume 0404_mae2d_small_X4/checkpoint-best.pth

