# rectangle-graspnet-multiObject-multiGrasp

### Introduction
**rectangle-graspnet-multiObject-multiGrasp** is a modified version of [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp) by [fujenchu](https://github.com/fujenchu). We have made some adjustment to the original code in order to apply it to the [graspnet](https://github.com/Fang-Haoshu/graspnetAPI) dataset.

###  Acknowledgement

The code of this repo is mainly based on [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp).

### Instructions

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

4. Download graspnet dataset from `https://graspnet.net/datasets.html`

5. Modify the paths of the dataset in the source code. More details in [doc/Path.md](doc/Path.md)

6. Create directories for processed data
```
cd data_process
mkdir grasp_data
cd grasp_data
mkdir Annotations
mkdir Images
mkdir ImageSets
```

7. Run data processing script
```
cd ../script
python data_preprocessing.py
cd ..
```

8. Move the processed data
```
mv grasp_data ../grasp_multiObject_multiGrasp/
cd ..
```

9. Train the model
```
cd grasp_multiObject_multiGrasp
./experiments/scripts/train_faster_rcnn.sh 1 graspRGB res50
```