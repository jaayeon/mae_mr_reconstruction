#!/bin/bash
python sleep.py
python main_pretrain.py --model vit2d_large --dataset fastmri --input_size 320 --batch_size 16 --epochs 100
python main_pretrain.py --model mae2d_large --dataset fastmri --input_size 320 --batch_size 16 --epochs 100
