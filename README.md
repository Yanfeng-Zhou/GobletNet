# GobletNet: Wavelet-Based High Frequency Fusion Network for Semantic Segmentation of Electron Microscopy Images

This is the official code of [GobletNet: Wavelet-Based High Frequency Fusion Network for Semantic Segmentation of Electron Microscopy Images](https://) (TMI 2024.10).

 ***Using the characteristics of  segmented images to drive the architecture design is the simplest but most effective！***

## EM Image Characteristics

We quantitatively analyze and summarize two characteristics of electron microscope (EM) images:
- **Characteristic 1** Compared with other images, the HF components of EM images based on the wavelet transform have richer texture details and clearer object contours but also have more noise.
- **Characteristic 2** For EM images, appropriately adding LF components to HF images can alleviate noise interference while maintaining sufficient HF details.

<table> <tr> <td align="center"> <img src="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/figure/Characteristic%201.png" alt="Image 1" width="100%"/><br/> Quantitative comparison of the HF information richness (HFIR), noise intensity (NI), and detail richness (DR) of datasets with different application scenarios, including natural, medical, microscopic and EM datasets. </td> <td align="center"> <img src="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/figure/Characteristic%202.png" alt="Image 2" width="100%"/><br/> Quantitative comparison of the HF information richness (HFIR), noise intensity (NI), and detail richness (DR) of EM datasets with different LF weights. &nbsp &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp</td> </tr> </table>

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/figure/Qualitative%20comparison%20of%20HF%20characteristics%20among%20natural%2C%20medical%2C%20microscopic%20and%20EM%20images.png" width="100%" >
<br>Qualitative comparison of HF characteristics among natural, medical, microscopic and EM images. (a) Raw images. (b) Wavelet transform results. (c) HF images. (d) Information richness heatmaps. (e) Noise intensity heatmaps. (f) Detailed distribution heatmaps. (g) Detailed distribution heatmaps (overlaid on raw images). (h) Ground truth.
</p>


## GobletNet
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/figure/Overview.png" width="100%" >
</p>

- For **Characteristic 1**, we use the HF image as an extra input and use an extra encoder to extract the rich HF information in HF image. 
- For **Characteristic 2**, we add LF components to HF image at a certain ratio to reduce the negative impact of excessive noise on model training.


## Quantitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/figure/Comparison%20results%20on%20EPFL%2C%20%20CREMI%2C%20SNEMI3D%20and%20UroCell.png" width="100%" >
</p>
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/figure/Comparison%20results%20on%20MitoEM%2C%20Nanowire%20and%20BetaSeg.png" width="100%" >
</p>

## Qualitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/GobletNet/blob/main/figure/Qualitative%20Comparison.png" width="100%" >
<br>(a) Raw images. (b) Ground truth. (c) SAM. (d) Deeplab V3+. (e) UNet 3+. (f) FusionNet. (g) WaveSNet. (h) UNet. (i) nnUNet. (j) GobletNet.
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
