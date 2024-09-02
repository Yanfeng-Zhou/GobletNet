# GobletNet: Wavelet-Based High Frequency Fusion Network for Semantic Segmentation of Electron Microscopy Images

This is the official code of [GobletNet: Wavelet-Based High Frequency Fusion Network for Semantic Segmentation of Electron Microscopy Images](https://) (TMI 2024.09).

## EM Image Characteristics




## GobletNet



## Quantitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/Semi-Supervision_2.png" width="100%" >
</p>

## Qualitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/Qualitative%20Comparison.png" width="100%" >
</p>

## Reimplemented Architecture
>We have reimplemented some semantic segmentation models with different application scenarios, including natural, medical, wavelet and EM models.
<table>
<tr><th align="left">Scenario</th> <th align="left">Model</th><th align="left">Code</th></tr>
<tr><td rowspan="3">Natural</td> <td>Deeplab V3+</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/deeplabv3.py">models/deeplabv3.py</a></td></tr>
<tr><td>Res-UNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/resunet.py">models/resunet.py</a></td></tr>
<tr><td>U<sup>2</sup>-Net</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/u2net.py">models/u2net.py</a></td></tr>
<tr><td rowspan="6">Medical</td><td>UNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/unet.py">models/unet.py</a></td></tr>
<tr><td>UNet++</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/unet_plusplus.py">models/unet_plusplus.py</a></td></tr>
<tr><td>Att-UNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/unet.py">models/unet.py</a></td></tr>
<tr><td>UNet 3+</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/unet_3plus.py">models/unet_3plus.py</a></td></tr>
<tr><td>SwinUNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/swinunet.py">models/swinunet.py</a></td></tr>
<tr><td>XNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/xnet.py">models/xnet.py</a></td></tr>
<tr><td rowspan="4">Wavelet</td><td>ALNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/aerial_lanenet.py">models/aerial_lanenet.py.py</a></td></tr>
<tr><td>MWCNN</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/mwcnn.py">models/mwcnn.py</a></td></tr>
<tr><td>WaveSNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/wavesnet.py">models/wavesnet.py</a></td></tr>
<tr><td>WDS</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/wds.py">models/wds.py</a></td>
</tr>
<tr><td rowspan="3">EM</td><td>DCR</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/dcr.py">models/dcr.py</a></td></tr>
<td>FusionNet</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/fusionnet.py">models/fusionnet.py</a></td></tr>
<tr><td>GobletNet (Ours)</td><td><a href="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/models/GobletNet.py">models/GobletNet.py</a></td></tr>
</table>

## Requirements
```
albumentations==1.2.1
einops==0.4.1
matplotlib==3.1.0
MedPy==0.4.0
numpy==1.21.6
opencv_python_headless==4.5.4.60
Pillow==10.4.0
PyWavelets==1.3.0
scikit_image==0.19.3
scikit_learn==1.5.1
scipy==1.7.3
SimpleITK==2.4.0
skimage==0.0
thop==0.1.1.post2209072238
timm==0.6.7
torch==1.8.0+cu111
torchio==0.18.84
torchvision==0.9.0+cu111
tqdm==4.64.0
tqdm_pathos==0.4
visdom==0.1.8.9
```

## Usage
- **Dataset preparation**
>Use [/tools/wavelet.py](https://github.com/Yanfeng-Zhou/GobletNet/blob/main/tools/wavelet.py) to generate wavelet transform results.
>Build your own dataset and its directory tree should be look like this:
```
dataset
├── train
    ├── image
        ├── 1.tif
        ├── 2.tif
        └── ...
    ├── H_0.1_db2
        ├── 1.tif
        ├── 2.tif
        └── ...
    └── mask
        ├── 1.tif
        ├── 2.tif
        └── ...
└── val
    ├── image
    └── mask
```

- **Configure dataset parameters**
>Add configuration in [/config/dataset_config/dataset_config.py](https://github.com/Yanfeng-Zhou/GobletNet/blob/main/config/dataset_config/dataset_cfg.py)
>The configuration should be as follows：
>
```
'CREMI':
	{
		'IN_CHANNELS': 1,
		'NUM_CLASSES': 2,
		'SIZE': (128, 128),
		'MEAN': [0.503902],
		'STD': [0.110739],
		'MEAN_H_0.1_db2': [0.515329],
		'STD_H_0.1_db2': [0.118728],
		'PALETTE': list(np.array([
			[255, 255, 255],
			[0, 0, 0],
		]).flatten())
	},
```

- **Training**
```
python -m torch.distributed.launch --nproc_per_node=4 train_GobletNet.py
```
- **Testing**
```
python -m torch.distributed.launch --nproc_per_node=4 test_GobletNet.py
```

## Citation
>If our work is useful for your research, please cite our paper:
```
```
