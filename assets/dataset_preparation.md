# Dataset preparation

First, create a folder named "benchmarks", wherever you want, and put its path in the .env file.

### UCM
This dataset has two versions, one with labels and one with captions. The following instructions prepare both of them.

1. Download the file ``UCM_captions.zip`` from [here](https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA).
2. Extract the zip file and open its content.
3. Create a folder named "UCM" inside ``benchmarks/``.
4. Rename the file ``dataset.json`` into ``captions.json`` and put it under ``benchmarks/UCM/labels/``.
5. Extract ``imgs.rar`` and put its content into ``benchmarks/UCM/images/captions/``.
6. Download the file ``UCMerced_LandUse.zip`` from [here](http://weegee.vision.ucmerced.edu/datasets/landuse.html).
7. Extract the zip file and open its content.
8. Copy the content of the folder ``Images`` into ``benchmarks/UCM/images/labels/``.
9. Copy the files ``metadata/UCM/UCM_captions.csv`` and ``metadata/UCM/UCM.csv`` into ``benchmarks/UCM/labels/``.

Final folder structure
```
| benchmarks
│  ├── UCM
│  │  ├── images
│  │  │  ├── captions
│  │  │  │  ├── 0.tif
│  │  │  |  ├── 1.tif
│  │  │  |  ├── ...
│  │  │  ├── labels
|  |  |  |  ├── agricultural
|  |  |  |  |  ├── agricultural_00.tif
|  |  |  |  |  ├── agricultural_01.tif
|  |  |  |  |  ├── ...
|  |  |  |  ├── baseball_diamond
|  |  |  |  |  ├── baseball_diamond_00.tif
|  |  |  |  |  ├── baseball_diamond_01.tif
│  │  ├── labels
│  │  │  ├── captions.json
│  │  │  ├── UCM_captions.csv
│  │  │  ├── UCM.csv
```

### WHU_RS19
1. Download the file ``WHU-RS19.zip`` from [here](https://captain-whu.github.io/BED4RS/#).
2. Create a folder named "WHU_RS19" inside ``benchmarks/``.
3. Extract the zip file and copy its content into ``benchmarks/WHU_RS19/images/``.
4. Copy the file ``metadata/WHU_RS19/WHU_RS19.csv`` into ``benchmarks/WHU_RS19/labels/``.

Final folder structure
```
| benchmarks
│  ├── WHU_RS19
│  │  ├── images
│  │  │  ├── airport
│  │  │  │  ├── airport_01.jpg
│  │  │  |  ├── airport_02.jpg
│  │  │  |  ├── ...
│  │  │  ├── beach
│  │  │  │  ├── beach-01.jpg
│  │  │  |  ├── beach-02.jpg
│  │  ├── labels
│  │  │  ├── WHU_RS19.csv
```

### RSSCN7
Steps:
1. Create a folder named "RSSCN7" inside ``benchmarks/`` and navigate to it.
2. Clone the repository inside the folder.
    ```bash
    git clone https://github.com/palewithout/RSSCN7
    ```
3. Copy the file ``metadata/RSSCN7/RSSCN7.csv`` inside ``benchmarks/RSSCN7/labels``.

This dataset does not provide train-test-val splits in literature. We created random train-test-val splits using stratification, to ensure that the classes are balanced in each split.

```
| benchmarks
│  ├── RSSCN7
│  │  ├── aGrass
│  │  │  ├── a001.jpg
│  │  │  ├── a002.jpg
│  │  │  ├── ...
│  │  ├── bField
│  │  │  ├── b001.jpg
│  │  │  ├── b002.jpg
│  │  │  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── RSSCN7.csv
```

### SIRI_WHU
1. Download the dataset from [here](https://www.kaggle.com/datasets/lzsy0226/siri-whu-data-set).
2. Create a folder named "SIRI_WHU" inside ``benchmarks/``.
3. Extract the zip file and copy all the images in ``benchmarks/SIRI_WHU/images/``.
4. Copy the file ``metadata/SIRI_WHU/SIRI_WHU.csv`` into ``benchmarks/SIRI_WHU/labels/``.

```
| benchmarks
│  ├── SIRI_WHU
│  │  ├── images
│  │  │  ├── agriculture
│  │  │  │  ├── 1.tif
│  │  │  |  ├── 2.tif
│  │  │  |  ├── ...
│  │  │  ├── commercial
│  │  │  │  ├── 1.tif
│  │  │  |  ├── 2.tif
│  │  │  |  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── SIRI_WHU.csv
```

### RESISC45
1. Download the file ``NWPU_RESISC45-20210923T210241Z-001.zip`` from [here](https://figshare.com/articles/dataset/NWPU-RESISC45_Dataset_with_12_classes/16674166).
2. Extract the zip file and copy its content into ``dataset/benchmarks/RESISC45/images/``.
3. Prepare the dataset using ``dataset/benchmarks/RESISC45/build_labels.ipynb``.

### RSI_CB256
1. Download the file ``RSI-CB256.rar`` from [here](https://github.com/lehaifeng/RSI-CB).
2. Extract the rar file and copy its content into ``dataset/benchmarks/RSI_CB256/images/``.
3. Prepare the dataset using ``dataset/benchmarks/RSI_CB256/build_labels.ipynb``.

### EuroSAT
1. Download the file ``EuroSAT.zip`` from [here](https://github.com/phelber/eurosat).
2. Extract the zip file and copy its content into ``dataset/benchmarks/EuroSAT/images/``.
3. Prepare the dataset using ``dataset/benchmarks/EuroSAT/build_labels.ipynb``.

### PatternNet
1. Download the file ``PatternNet.zip`` from [here](https://sites.google.com/view/zhouwx/dataset).
2. Extract the zip file and open its content.
3. Copy the content of the folder ``images`` into ``dataset/benchmarks/PatternNet/images/``.
4. Prepare the dataset using ``dataset/benchmarks/PatternNet/build_labels.ipynb``.

### OPTIMAL_31
1. Download the file ``archive.zip`` from [here](https://www.kaggle.com/datasets/brajrajnagar/optimal-31).
2. Extract the zip file and open its content.
3. Copy the content of the folder ``Images`` into ``dataset/benchmarks/OPTIMAL_31/images/``.
4. Prepare the dataset using ``dataset/benchmarks/OPTIMAL_31/build_labels.ipynb``.

### MLRSNet
1. Download the file ``MLRSNet A Multi-label High Spatial Resolution Remote Sensing Dataset for Semantic Scene Understanding.zip`` from [here](https://github.com/cugbrs/MLRSNet).
2. Extract the zip file and open its content.
3. Copy the content of the folder ``Labels`` into ``dataset/benchmarks/MLRSNet/labels/data/``.
4. Extract each rar file of the folder ``Images`` into ``dataset/benchmarks/MLRSNet/images/``.
5. Prepare the dataset using ``dataset/benchmarks/MLRSNet/build_labels.ipynb``.

### RSICD
1. Download the file ``RSICD_optimal-master.zip`` from [here](https://github.com/201528014227051/RSICD_optimal).
2. Extract the zip file and open its content.
3. Copy the file ``dataset_rsicd.json`` into ``dataset/benchmarks/RSICD/labels/sentences/``.
4. Extract the file ``txtclasses_rsicd.rar`` and copy its content into ``dataset/benchmarks/RSICD/labels/classes/``.
5. Extract the file ``RSICD_images.zip`` and copy its content into ``dataset/benchmarks/RSICD/images/``.
6. Prepare the dataset using ``dataset/benchmarks/RSICD/build_labels.ipynb``.

### RSITMD
1. Download the file ``RSITMD.zip`` from [here](https://github.com/xiaoyuan1996/AMFMN).
2. Extract the zip file and open its content.
3. Copy the content of the folder ``images`` into ``dataset/benchmarks/RSITMD/images/``.
5. Copy the file ``dataset_RSITMD.json`` into ``dataset/benchmarks/RSITMD/labels/``.
4. Prepare the dataset using ``dataset/benchmarks/RSITMD/build_labels.ipynb``.

### SIDNEY

### BIBTEX
```
    @inproceedings{Radford2021LearningTV,
        title={Learning Transferable Visual Models From Natural Language Supervision},
        author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
        booktitle={International Conference on Machine Learning},
        year={2021},
        url={https://api.semanticscholar.org/CorpusID:231591445}
        }
```