# McByte installation

The installation is pretty straightforward. For the smooth experience, please follow the exact order of the instructions.


## Versioning

This installation has been tested and works well on Linux with <b>CUDA 11.6</b> and <b>GCC 10.4</b>.<br/>
Please have these two installed and enabled/loaded first.<br/>
We also recommend using the same versions of the other packages and libraries as specified below.


## Virtual environment

Install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install). Based on your needs and resources (e.g. remote server), consider [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) instead.


Create and activate a conda environment with Python as specified:
```
conda create -n mcbyte python=3.9
conda activate mcbyte
```


## Main tracker installation

Clone the repository into your folder and go to McByte's root:
```
git clone https://github.com/tstanczyk95/McByte.git
cd McByte
```

Install PyTorch and related packages as specified below. Command taken from [here](https://pytorch.org/get-started/previous-versions/). 
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Install the required packages following the instructions:
```
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
pip install --upgrade numpy==1.23
```
<i>(In case see an error in red about pip's dependency resolver, do not worry about it, you will be able to proceed and still finish the installation and run Mcbyte)</i>

Make directory for the pretrained detector ([YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), already installed) models:
```
mkdir pretrained
```

Download the following pre-trained models from their original sources:
- [SportsMOT](https://github.com/MCG-NJU/MixSort?tab=readme-ov-file#model-zoo): yolox_x_sports_mix.pth.tar 🔥 <b> Recommended for the demo in your sport setting frames/videos </b> 🔥
- [DanceTrack](https://huggingface.co/noahcao/dancetrack_models/tree/main/bytetrack_models): bytetrack_model.pth.tar
- [MOT17](https://github.com/FoundationVision/ByteTrack?tab=readme-ov-file#model-zoo): bytetrack_x_mot17.pth.tar and bytetrack_ablation.pth.tar

Place them in the <i>pretrained</i> folder.<br/>
If you cannot reach some of these models, then please see the last section of this page.


## Mask propagation installation

Install the temporally propagated segmentation mask functionality ([Cutie](https://github.com/hkchengrex/Cutie/tree/main)) as follows:

```
cd mask_propagation/Cutie
pip install -e .
```

Download the model weights
```
mkdir weights
cd weights
wget https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth
wget https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth # [optional]
cd ..
```
Otherwise, run: 
```
python cutie/utils/download_models.py
```
If you cannot reach some of these models, then please see the last section of this page.

## Image segmentation mask installation

Install the image segmentation mask functionality ([SAM](https://github.com/facebookresearch/segment-anything)) as follows:
```
cd ../.. # Ensure you are in the McByte main folder now
pip install git+https://github.com/facebookresearch/segment-anything.git
mkdir sam_models
cd sam_models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..
```
If you cannot reach this model, then please see the last section of this page.

## Error handling
### Numpy still as >= 2.0
If running McByte gives you an error about Numpy 2.0 or higher, run this line one more time:
```
 pip install --upgrade numpy==1.23
```
It might happen that Cutie or SAM installation updates this version by default, after running this line for the first time above, hence you need to run it again. (The same remark with the error in red applies here.)
### Unrecognized function arguments
If you modify function signatures or pull an updated version, and it will result in an error of unrecognized arguments, please run the setup install instruction again in the main McByte folder:
```
python3 setup.py develop
```
It will adjsut the running settings to the new signatures, etc.

## Pretrained model backups.

In case the original models links are broken and they cannot be downloaded, you can also reached them [here](https://drive.google.com/drive/folders/1yzzJk9dpJUY3lIHdkkFyGtKL2F-FenN6?usp=sharing). The following models are included: <br/>
- YOLOX SportsMOT: yolox_x_sports_mix.pth.tar
- YOLOX DanceTrack: bytetrack_x_dancetrack.pth.tar
- YOLOX MOT17: bytetrack_x_mot17.pth.tar, bytetrack_ablation.pth.tar
- Cutie base mega: cutie-base-mega.pth
- Cutie lvis h18: coco_lvis_h18_itermask.pth
- SAM vit b: sam_vit_b_01ec64.pth
<br/>

(Credit belongs to the original authors, see the references in the sections above.)
