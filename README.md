# rectangle-graspnet-multiObject-multiGrasp

### Introduction
**rectangle-graspnet-multiObject-multiGrasp** is a modified version of [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp) by [fujenchu](https://github.com/fujenchu). We have made some adjustment to the original code in order to apply it to the [graspnet](https://github.com/Fang-Haoshu/graspnetAPI) dataset.

###  Acknowledgement

The code of this repo is mainly based on [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp).

### Install

1. Clone the code
```
git clone https://github.com/graspnet/rectangle-graspnet-multiObject-multiGrasp
cd rectangle-graspnet-multiObject-multiGrasp/grasp_multiObject_multiGrasp
```

2. Prepare environment (Need Anaconda or Miniconda)
```
conda env create -f grasp_env.yaml
conda activate grasp
```

3. Build Cython modules
```
cd lib
make clean
make
cd ..
```

4. Install [Python COCO API](https://github.com/cocodataset/cocoapi)
```
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../../..
```

### Graspnet Dataset

```
mkdir graspnet_dataset
```

Then download the graspnet dataset from `https://graspnet.net/datasets.html`

- Move the dataset to [./graspnet_dataset](./graspnet_dataset)

- Or you can link the path of the graspnet dataset to [./graspnet_dataset](./graspnet_dataset) by

  ```
  ln -s /path/to/graspnet ./graspnet_dataset
  ```

- Or you can modify GRASPNET_ROOT in [grasp_multiObject_multiGrasp/tools/graspnet_config.py](grasp_multiObject_multiGrasp/tools/graspnet_config.py) directly

**NOTICE:** Your path should match the following structure details

```
graspnet_dataset
|-- scenes
    |-- scene_0000
    |   |-- object_id_list.txt
    |   |-- rs_wrt_kn.npy
    |   |-- kinect
    |   |   |-- rgb
    |   |   |   |-- 0000.png to 0255.png
    |   |   `-- depth
    |   |   |   |-- 0000.png to 0255.png
    |   |   `-- label
    |   |   |   |-- 0000.png to 0255.png
    |   |   `-- annotations
    |   |   |   |-- 0000.xml to 0255.xml
    |   |   `-- meta
    |   |   |   |-- 0000.mat to 0255.mat
    |   |   `-- rect
    |   |   |   |-- 0000.npy to 0255.npy
    |   |   `-- camK.npy
    |   |   `-- camera_poses.npy
    |   |   `-- cam0_wrt_table.npy
    |   |
    |   `-- realsense
    |       |-- same structure as kinect
    |
    |
    `-- scene_0001
    |
    `-- ... ...
    |
    `-- scene_0189
```



### Demo

1. Download pretrained models

   - Download the model from [Google Drive](https://drive.google.com/file/d/1QrjLDKr8eHgN0rM48YpWXY-sN89zJNim/view?usp=sharing), or [JBOX](https://jbox.sjtu.edu.cn/l/J5z6gE), or [Baidu Pan (Password: v9j7)](https://pan.baidu.com/s/19Vp8DWbpFdDQfeICVwRTew)
   - Move it to [grasp_multiObject_multiGrasp/output/res50/train/default/](grasp_multiObject_multiGrasp/output/res50/train/default/)

2. Run demo

   ```
   cd grasp_multiObject_multiGrasp/tools
   python demo_graspRGD.py --net res50 --dataset grasp
   cd ../..
   ```

### Data Preprocessing

1. Choose the type of the camera by changing CAMERA_NAME(line 2) in [grasp_multiObject_multiGrasp/tools/graspnet_config.py](grasp_multiObject_multiGrasp/tools/graspnet_config.py)
2. Run data processing script
```
cd data_process/script
python data_preprocessing.py
cd ..
```

3. Move the processed data

```
mv grasp_data ../grasp_multiObject_multiGrasp/
cd ..
```

### Training

1. Download the `res50` pretrained model

   - Download the model from [Google Drive](https://drive.google.com/file/d/1srBA9KQZJnuI59kZUZ7lGWWN6bcTzVNC/view?usp=sharing), or [JBOX](https://jbox.sjtu.edu.cn/l/Vooj01), or [Baidu Pan (Password: tl84)](https://pan.baidu.com/s/1PhGBwKVd5o0Q9qNIy8AI4Q)

   - Move the `res50.ckpt` file to [grasp_multiObject_multiGrasp/data/imagenet_weights/](grasp_multiObject_multiGrasp/data/imagenet_weights/)

2. If you have stored the pretrained models in [grasp_multiObject_multiGrasp/output/res50/train/default/](grasp_multiObject_multiGrasp/output/res50/train/default/)

   - Make sure there's nothing in [grasp_multiObject_multiGrasp/output/res50/train/default/](grasp_multiObject_multiGrasp/output/res50/train/default/)

   - You can rename the directory. For example:

     ```
     mv grasp_multiObject_multiGrasp/output/res50 grasp_multiObject_multiGrasp/output/res50_pretrained
     ```

   - Or you can move the directory [grasp_multiObject_multiGrasp/output/res50/](grasp_multiObject_multiGrasp/output/res50/) to somewhere else

3. Training
```
cd grasp_multiObject_multiGrasp
./experiments/scripts/train_faster_rcnn.sh 0 graspRGB res50
```