# PointNGCNN: Deep convolutional networks on 3D point clouds with neighborhood graph filters
## Installation
Install [Tensorflow](https://www.tensorflow.org/install/). You may also need to install h5py. The code has been tested with python 3.5, tensorflow 1.7, CUDA 9.0 and cuDNN 7.0 on Ubuntu 16.04.
## Usage
We used the same data in this paper as [pointnet++](https://github.com/charlesq34/pointnet2). Please go to data folder, follow instructions there and download the data. You are recommended to put unziped modelnet and shapenet folders in data/, but they can of course go somewhere else.
### - Shape Classification
#### * Run the training script:
``` python train.py ```
#### * Run the evaluation script after training finished:
``` python evalutate.py ```
### - Part Segmentation
Follow ModelNet instructions, but apply those in folder "part_seg", and you should be able to run the experiments smoothly.


