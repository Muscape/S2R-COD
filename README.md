# S2R-COD: Synthetic-to-Real Camouflaged Object Detection
This is the official (Pytorch) implementation for the paper "Synthetic-to-Real Camouflaged Object Detection", ACM MM 2025.


# 🛠️Setup
## Runtime

The main python libraries we use:
- Python 3.10.11
- torch 2.0.1
- prettytable 3.16.0

## Datasets
Please create a directory named `Dataset` in current directory, then download the following datasets and unzip into `Dataset`:


Source (Synthetic):
- HKU-IS ( [GoogleDrive](https://drive.google.com/file/d/10fyub8eQ4QKnpW_tK9zWyMiWept-ZO0m/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/1SKs1yCjCNsW2q__S58pRKg?pwd=ed3r) )
- CAMO + NC4K + CHAMELEON(CNC) ( [GoogleDrive](https://drive.google.com/file/d/1ZqBaxC72LLqBQHpQ3DyfHR7rePy6UFsx/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/1jmJLqEso8T56mZWTmxbYrA?pwd=xwvg) )

Target (Real):
- COD10K-train ( [GoogleDrive](https://drive.google.com/file/d/1KCyif8Pe5MrmBxxdj5Dp-zo8jZAaKVoK/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/1xTm3Q5kPoaP2rEdHe95cwA?pwd=xgbb) )

Test (Real): 
- COD10K-test ( [GoogleDrive](https://drive.google.com/file/d/128sXPCAfRgFPXOqv3-uY6WPGLMfF27Fh/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/1I7leLjvsXaU_kET9m-7Vuw?pwd=pq73) )

Val (Real):
- CAMO ( [GoogleDrive](https://drive.google.com/file/d/183R3IviOU6KlycQCx12kp6IGJXWoP-_9/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/1z-lynX0feInM_ISVC3sZPg?pwd=hd6u) )

# 🎢Run
After finishing above steps, your directory structure of code may like this:
```text
S2R-COD/
    |-- Dataset/
        |-- Source/
            |-- CNC/
            |-- HKU-IS/
        |-- Target/
        |-- Test/
        |-- Val/
    |-- Eval/
    |-- Src/
    CLS.py
    MyTest.py
    MyTrain.py
    README.md
```

## Training

- Downloading ResNet wights ( [GoogleDrive](https://drive.google.com/file/d/1At5pec341s0ZAj_ihFaxDjwrPDDnroBH/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/1-KhUqImRCCpxcUsWpJul5w?pwd=bds9) ) on ImageNet dataset for SINet, and move it into `./Src/model/SINet`. Res2Net ( [GoogleDrive](https://drive.google.com/file/d/1tyryURHR3lNLVJ1Dv1m_7scoi92jHgl6/view?usp=drive_link) | [BaiduYun](https://pan.baidu.com/s/18EL_DPhcjEfSilLrwmswLA?pwd=vqiy) ) for SINet-v2 and move it into `./Src/model/SINetV2`.

- You can run the training script using the following command:

```text
python Mytrain.py --network [SINet, SINet-v2] --task [C2C, S2C] --save_model Your_save_path --source_root Your_source_path
```

## Testing
```text
python Mytest.py --network [SINet, SINet-v2] --model_path Your_checkpoint_path --test_save Your_mask_path
```
*(\[x,y,z] means choices.)*

## Evalution
We adopt the evaluation protocol from the [DGNet](https://github.com/GewelsJI/DGNet/tree/main) project.
```text
python MyEval.py --pred_root Your_mask_path --txt_name Your_result_path
```

## Pretrained models
-C2C: SINet ( [GoogleDrive]() | [BaiduYun]() ) SINet-v2 ( [GoogleDrive]() | [BaiduYun]() )

-S2C: SINet ( [GoogleDrive]() | [BaiduYun]() ) SINet-v2 ( [GoogleDrive]() | [BaiduYun]() )


# ⚖️License
This source code is released under the MIT license. View it [here](LICENSE)