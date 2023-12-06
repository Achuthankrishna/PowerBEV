# PowerBEV - Woven Dataset



![](.github/PowerBev2.jpg)

## 📃 Contents
- [PowerBEV2](#powerbev)
  - [Setup](#️-setup)
  - [Dataset](#-dataset)
  - [Pre-trained models](#-pre-trained-models)
  - [Training](#-training)
  - [Prediction](#-prediction)
    - [Evaluation](#evaluation)
    - [Visualisation](#visualisation)
  - [Credits](#-license)
  - [Citation](#-citation)

## Setup
Create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment by running 
```
conda env create -f environment.yml
```

## Dataset
- Download the full [*Toyota Woven Planet Perception datset*](https://woven.toyota/en/perception-dataset/), which includes the *Mini dataset* and the *Train and Test dataset*.
- Extract the tar files to a directory named `lyft2/` . The files should be organized in the following structure:
  ```
  lyft2/
  ├──── train/
  │     ├──── maps/
  │     ├──── images/
  │     ├──── train_lidar/
  │     └──── train_data/
  ```

## Pre-trained models (Comparision)
The config file can be found in [`powerbev/configs`](powerbev/configs) . You can download the pre-trained models which are finetuned for nuscenes dataset using the below links:

|Weights | Dataset | BEV Size | IoU | VPQ |
|-|-|-|:-:|:-:|
|[`PowerBEV_long.ckpt`](https://drive.google.com/file/d/1P33nD6nt8IjnvKTd4WlTKWbarFdCE34f/view?usp=sharing) | NuScenes| 100m x 100m (50cm res.) | 39.3 | 33.8 |
| [`PowerBEV_short.ckpt`](https://drive.google.com/file/d/1-T4R6vC2HHhqxXeUeUg-CuViA5XdQEcV/view?usp=sharing) | NuScenes| 30m x 30m (15cm res.) | 62.5 | 55.5 |  
| [`PowerBEV_short.ckpt`]((https://drive.google.com/file/d/16bnG3kI_J3JkFGGxMuQfz879QFz7SVhj/view?usp=sharing))| None | 100m x 100m (50cm res.) | 39.3 | 33.8 |
| [`PowerBEV_static_short.ckpt`](https://drive.google.com/file/d/1Jwb2UjNEuamwNmBZ_R-DAW91dhxi4_6J/view?usp=sharing)| None | 30m x 30m (15cm res.) | 62.5 | 55.5 |  

## 🏊 Training
To train the model from scratch on NuScenes, run

```
python train.py --config powerbev/configs/powerbev.yml
```

To train the model from the pre-trained static checkpoint on NuScenes, download pre-trained static weights ([`static long`](https://drive.google.com/file/d/16bnG3kI_J3JkFGGxMuQfz879QFz7SVhj/view?usp=sharing)/[`static short`](https://drive.google.com/file/d/1Jwb2UjNEuamwNmBZ_R-DAW91dhxi4_6J/view?usp=sharing)) to `YOUR_PRETRAINED_STATIC_WEIGHTS_PATH` and run

```
python train.py --config powerbev/configs/powerbev.yml \
                PRETRAINED.LOAD_WEIGHTS True \
                PRETRAINED.PATH $YOUR_PRETRAINED_STATIC_WEIGHTS_PATH
```

Note: These will train the model on 4 GPUs, each with a batch of size 2. 

To set your configs, please run

```
python train.py --config powerbev/configs/powerbev.yml \
                DATASET.DATAROOT $YOUR_NUSCENES_DATAROOT \
                LOG_DIR $YOUR_OUTPUT_PATH \
                GPUS [0] \
                BATCHSIZE $YOUR_DESIRED_BATCHSIZE
```

The above settings can also be changed directly by modifying [`powerbev.yml`](powerbev/configs/powerbev.yml). Please see the [`config.py`](powerbev/config.py) for more information.  

## 🏄 Prediction
### Evaluation
Download trained weights ([`long`](https://drive.google.com/file/d/1P33nD6nt8IjnvKTd4WlTKWbarFdCE34f/view?usp=sharing)/[`short`](https://drive.google.com/file/d/1-T4R6vC2HHhqxXeUeUg-CuViA5XdQEcV/view?usp=sharing)) to `YOUR_PRETRAINED_WEIGHTS_PATH` and run
```
python test.py --config powerbev/configs/powerbev.yml \
                PRETRAINED.LOAD_WEIGHTS True \
                PRETRAINED.PATH $YOUR_PRETRAINED_WEIGHTS_PATH
```

### Visualisation
Download trained weights ([`long`](https://drive.google.com/file/d/1P33nD6nt8IjnvKTd4WlTKWbarFdCE34f/view?usp=sharing)/[`short`](https://drive.google.com/file/d/1-T4R6vC2HHhqxXeUeUg-CuViA5XdQEcV/view?usp=sharing)) to `YOUR_PRETRAINED_WEIGHTS_PATH` and run
```
python visualise.py --config powerbev/configs/powerbev.yml \
                PRETRAINED.LOAD_WEIGHTS True \
                PRETRAINED.PATH $YOUR_PRETRAINED_WEIGHTS_PATH \
                BATCHSIZE 1
```
This will render predictions from the network and save them to an `visualization_outputs` folder.
Note: To visualize Ground Truth, please add the config `VISUALIZATION.VIS_GT True` at the end of the command

## 📜 License
PowerBEV is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Credits
This is the official PyTorch implementation of the paper: 
> [**PowerBEV: A Powerful yet Lightweight Framework for Instance Prediction in Bird's-Eye View**](https://www.ijcai.org/proceedings/2023/0120.pdf)  
> Peizheng Li, Shuxiao Ding,Xieyuanli Chen,Niklas Hanselmann,Marius Cordts,Jürgen Gall

## 🔗 Citation
```
@article{li2023powerbev,
  title     = {PowerBEV: A Powerful Yet Lightweight Framework for Instance Prediction in Bird's-Eye View},
  author    = {Li, Peizheng and Ding, Shuxiao and Chen, Xieyuanli and Hanselmann, Niklas and Cordts, Marius and Gall, Juergen},
  journal   = {arXiv preprint arXiv:2306.10761},
  year      = {2023}
}
@inproceedings{ijcai2023p120,
  title     = {PowerBEV: A Powerful Yet Lightweight Framework for Instance Prediction in Bird’s-Eye View},
  author    = {Li, Peizheng and Ding, Shuxiao and Chen, Xieyuanli and Hanselmann, Niklas and Cordts, Marius and Gall, Juergen},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {1080--1088},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/120},
  url       = {https://doi.org/10.24963/ijcai.2023/120},
}
```