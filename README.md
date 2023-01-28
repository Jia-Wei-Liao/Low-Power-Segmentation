# MediaTek LowPower Semantic Segmentation

## Getting the code
You can download all the files in this repository by cloning this repository:  
```
git clone https://github.com/Jia-Wei-Liao/MediaTek_LowPower_Semantic_Segmentation.git
```

## Set up enviroment
**1. Create a conda enviroment**
```
conda create -n MTKML python==3.7 -y
conda activate MTKML
```
**2. Install the requirement package**
```
pip install -r requirements.txt
conda install cudatoolkit=10.0
```


## Training
To train the model, you can run this command:
```
python train.py
```


## Quantization
To get the low power model, you can run this command:
```
python quantization.py
```

## Experiment
In the following figure, we show the result of raw image, raw mask, model predication and model predication after quantization.

![Image](https://github.com/Jia-Wei-Liao/MediaTek_LowPower_Semantic_Segmentation/blob/main/figure/image.jpg "raw image")
![Image](https://github.com/Jia-Wei-Liao/MediaTek_LowPower_Semantic_Segmentation/blob/main/figure/mask.jpg  "raw mask")
![Image](https://github.com/Jia-Wei-Liao/MediaTek_LowPower_Semantic_Segmentation/blob/main/figure/prediction.jpg "model prediction w/o quantization")
![Image](https://github.com/Jia-Wei-Liao/MediaTek_LowPower_Semantic_Segmentation/blob/main/figure/prediction_quantization.jpg "model prediction w/ quantization")

<table>
  <tr>
    <td>metric</td>
    <td>value</td>
    <td>rank</td>
  </tr>
  <tr>
    <td>latency</td>
    <td>90896.9</td>
    <td>5/55</td>
  </tr>
  <tr>
    <td>power</td>
    <td>689.76</td>
    <td>31/55</td>
  </tr>
  <tr>
    <td>mIoU</td>
    <td>0.5672</td>
    <td>22/55</td>
  </tr>
  <tr>
    <td>total</td>
    <td>86.25</td>
    <td>9/55</td>
  </tr>
<table>

## Citation
```
@misc{
    title  = {mediaTek_low_power_semantic_segmentation},
    author = {Jia-Wei Liao},
    url    = {https://github.com/Jia-Wei-Liao/MediaTek_LowPower_Semantic_Segmentation},
    year   = {2022}
}
```
