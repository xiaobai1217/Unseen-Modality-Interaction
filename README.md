## [Learning Unseen Modality Interaction](https://arxiv.org/abs/2306.12795) (NeurIPS 2023)

Yunhua Zhang, Hazel Doughty, Cees G.M. Snoek

This is the demo code for the video classification task using EPIC-Kitchens, with RGB and audio modalities. 


### Environment
* Python 3.8.5
* torch 1.12.1+cu113
* torchaudio 0.12.1+cu113
* torchvision 0.13.1+cu113
* mmcv-full 1.7.0

We provide the splits for training, validation and testing in the `epic-annotations` folder. 

To run the code:
`python train.py --lr 1e-1 --batch_size 96 --save_name 1e-1`

We finetuned the model by reduced learning rates, as specified in `bash.sh`.
