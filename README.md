# ML_IDCard_Segmentation (IN WORK)
Machine Learning Project to identify an ID Card on an image.

## Aditional Information
Dataset: [MIDV-500](https://arxiv.org/abs/1807.05786)   
Tensorflow Version: GPU 1.5.0

### Installation
1. Create and activate a new environment.
```
conda create -n idcard python=3.6
source activate idcard
```
2. Install Dependencies.
```
pip install -r requirements.txt
```

### Download and Prepare Dataset
```
python prepare_dataset.py
```

### Train
```
python train.py
```
