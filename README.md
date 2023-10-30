# 1. Overview
This repository is an exercise in image segmentation using lung images as the dataset. 

Medical Image Segmentation is the process of automatic detection of boundaries within images. In this exercise, I train a convolutional neural network with [U-Net](https://arxiv.org/abs/1505.04597) architecture.

Inspiration was taken from this notebook: 
https://www.kaggle.com/code/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen

# 2. Data Preparation

1. Combine left and right lung segmentation masks of Montgomery chest x-rays
1. Resize images to 512x512 pixels
1. Split images into training and test datasets
1. Write images to /segmentation directory