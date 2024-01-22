#!/bin/bash
python train.py --lr 1e-1 --batch_size 96 --save_name 1e-1
python train.py --lr 1e-2 --batch_size 96 --save_name 1e-2resumed --resume_training True --resume_checkpoint checkpoints/best_multimodal1e-1.pt
python train.py --lr 1e-3 --batch_size 96 --save_name 1e-3resumed --resume_training True --resume_checkpoint checkpoints/best_multimodal1e-2resumed.pt