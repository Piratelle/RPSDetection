# Rock, Paper, Scissors Detection
This project aims to use [OpenMMLab tools](https://github.com/open-mmlab) to detect and recognize the specific gestures used in Rock, Paper, Scissors within two-dimensional images. Developed as a part of the **Computer Vision** curriculum at SDSU for the Spring 2024 semester.

## Setup & Installation
In order to incorporate **mmpose**, we referenced the [mmpose Installation Guide](https://mmpose.readthedocs.io/en/latest/installation.html), with adaptations as required.

To set up the environment:
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

To install **pytorch**:
```
conda install pytorch torchvision -c pytorch
```

To install **mim** and the OpenMMLab packages:
```
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```