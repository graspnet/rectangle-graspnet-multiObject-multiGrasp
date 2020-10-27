# rectangle-graspnet-multiObject-multiGrasp

### Introduction
**rectangle-graspnet-multiObject-multiGrasp** is a modified version of [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp) by [fujenchu](https://github.com/fujenchu). We have made some adjustment to the original code in order to apply it to the [graspnet](https://github.com/Fang-Haoshu/graspnetAPI) dataset.

###  Acknowledgement

The code of this repo is mainly based on [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp).

### Environment

```
TODO
```

### Install

1. Clone the code
```
git clone https://github.com/graspnet/rectangle-graspnet-multiObject-multiGrasp
cd rectangle-graspnet-multiObject-multiGrasp/grasp_multiObject_multiGrasp
```

2. Build Cython modules
```
cd lib
make clean
make
cd ..
```

3. Install [Python COCO API](https://github.com/cocodataset/cocoapi)
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

### Demo

1. Download pretrained models

   - Download the model from  [TODO](TODO)
   - Move it to [grasp_multiObject_multiGrasp/output/res50/train/default/](grasp_multiObject_multiGrasp/output/res50/train/default/)

2. Run demo

   ```
   cd grasp_multiObject_multiGrasp/tools
   python demo_graspRGD.py --net res50 --dataset grasp
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

```
cd grasp_multiObject_multiGrasp
./experiments/scripts/train_faster_rcnn.sh 1 graspRGB res50
```