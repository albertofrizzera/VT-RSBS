# Dataset preparation

First, create a folder named "benchmarks", wherever you want, and put its path in the .env file.

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

Final folder structure
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

Final folder structure
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
1. Create a folder named "RESISC45" inside ``benchmarks/``.
2. Download the file ``NWPU_RESISC45-20210923T210241Z-001.zip`` from [here](https://figshare.com/articles/dataset/NWPU-RESISC45_Dataset_with_12_classes/16674166).
3. Extract the zip file and copy its content into ``benchmarks/RESISC45/images/``.
4. Copy the file ``metadata/RESISC45/RESISC45.csv`` into ``benchmarks/RESISC45/labels/``.

Final folder structure
```
| benchmarks
│  ├── RESISC45
│  │  ├── images
│  │  │  ├── airfield
│  │  │  │  ├── img0000.tif
│  │  │  |  ├── img0001.tif
│  │  │  |  ├── ...
│  │  │  ├── anchorage
│  │  │  │  ├── img0000.tif
│  │  │  |  ├── img0001.tif
│  │  │  |  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── RESISC45.csv
```

### RSI_CB128
1. Create a folder named "RSI_CB128" inside ``benchmarks/``.
2. Download the file ``RSI-CB128.rar`` from [here](https://github.com/lehaifeng/RSI-CB).
3. Extract the rar file and copy its content into ``benchmarks/RSI_CB128/images/``.
4. Copy the file ``metadata/RSI_CB128/RSI_CB128.csv`` into ``benchmarks/RSI_CB128/labels/``.

Final folder structure
```
| benchmarks
│  ├── RSI_CB128
│  │  ├── images
│  │  │  ├── construction_land
│  │  │  │  ├── city_building
│  │  │  │  │  ├── city_building_(1).tif
│  │  │  │  │  ├── city_building_(2).tif
│  │  │  │  │  ├── ...
│  │  │  │  ├── container
│  │  │  │  │  ├── container_(1).tif
│  │  │  │  │  ├── container_(2).tif
│  │  │  │  │  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── RSI_CB128.csv
```

### RSI_CB256
1. Create a folder named "RSI_CB256" inside ``benchmarks/``.
2. Download the file ``RSI-CB256.rar`` from [here](https://github.com/lehaifeng/RSI-CB).
3. Extract the rar file and copy its content into ``benchmarks/RSI_CB256/images/``.
4. Copy the file ``metadata/RSI_CB256/RSI_CB256.csv`` into ``benchmarks/RSI_CB256/labels/``.

Final folder structure
```
| benchmarks
│  ├── RSI_CB256
│  │  ├── images
│  │  │  ├── construction_land
│  │  │  │  ├── city_building
│  │  │  │  │  ├── city_building_(1).tif
│  │  │  │  │  ├── city_building_(2).tif
│  │  │  │  │  ├── ...
│  │  │  │  ├── container
│  │  │  │  │  ├── container_(1).tif
│  │  │  │  │  ├── container_(2).tif
│  │  │  │  │  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── RSI_CB256.csv
```

### EuroSAT
1. Create a folder named "EuroSAT" inside ``benchmarks/``.
2. Download the file ``EuroSAT.zip`` from [here](https://github.com/phelber/eurosat).
3. Extract the zip file and copy its content into ``benchmarks/EuroSAT/images/``.
4. Copy the file ``metadata/EuroSAT/EuroSAT.csv`` into ``benchmarks/EuroSAT/labels/``.

Final folder structure
```
| benchmarks
│  ├── EuroSAT
│  │  ├── images
│  │  │  ├── annual_crop
│  │  │  │  ├── AnnualCrop_1.jpg
│  │  │  |  ├── AnnualCrop_2.jpg
│  │  │  |  ├── ...
│  │  │  ├── forest
│  │  │  │  ├── Forest_1.jpg
│  │  │  |  ├── Forest_2.jpg
│  │  │  |  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── EuroSAT.csv
```

### PatternNet
1. Create a folder named "PatternNet" inside ``benchmarks/``.
2. Download the file ``PatternNet.zip`` from [here](https://sites.google.com/view/zhouwx/dataset).
3. Extract the zip file and open its content.
4. Copy the content of the folder ``images`` into ``benchmarks/PatternNet/images/``.
5. Copy the file ``metadata/PatternNet/PatternNet.csv`` into ``benchmarks/PatternNet/labels/``.

Final folder structure
```
| benchmarks
│  ├── PatternNet
│  │  ├── images
│  │  │  ├── airplane
│  │  │  │  ├── airplane001.jpg
│  │  │  |  ├── airplane002.jpg
│  │  │  |  ├── ...
│  │  │  ├── baseball_field
│  │  │  │  ├── baseballfield001.jpg
│  │  │  |  ├── baseballfield002.jpg
│  │  │  |  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── PatternNet.csv
```

### OPTIMAL_31
1. Create a folder named "OPTIMAL_31" inside ``benchmarks/``.
2. Download the file ``archive.zip`` from [here](https://www.kaggle.com/datasets/brajrajnagar/optimal-31).
3. Extract the zip file and open its content.
4. Copy the content of the folder ``Images`` into ``benchmarks/OPTIMAL_31/images/``.
5. Copy the file ``metadata/OPTIMAL_31/OPTIMAL_31.csv`` into ``benchmarks/OPTIMAL_31/labels/``.

Final folder structure
```
| benchmarks
│  ├── OPTIMAL_31
│  │  ├── images
│  │  │  ├── airplane
│  │  │  │  ├── airplane(1).jpg
│  │  │  |  ├── airplane(2).jpg
│  │  │  |  ├── ...
│  │  │  ├── airport
│  │  │  │  ├── airport(1).jpg
│  │  │  |  ├── airport(2).jpg
│  │  │  |  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── OPTIMAL_31.csv
```

### MLRSNet
1. Create a folder named "MLRSNet" inside ``benchmarks/``.
2. Download the file ``MLRSNet A Multi-label High Spatial Resolution Remote Sensing Dataset for Semantic Scene Understanding.zip`` from [here](https://github.com/cugbrs/MLRSNet).
3. Extract the zip file and open its content.
4. Copy the content of the folder ``Labels`` into ``benchmarks/MLRSNet/labels/data/``.
5. Extract each rar file of the folder ``Images`` into ``benchmarks/MLRSNet/images/``.
6. Copy the file ``metadata/MLRSNet/MLRSNet.csv`` into ``benchmarks/MLRSNet/labels/``.

Final folder structure
```
| benchmarks
│  ├── MLRSNet
│  │  ├── images
│  │  │  ├── airplane
│  │  │  │  ├── airplane_00001.jpg
│  │  │  |  ├── airplane_00002.jpg
│  │  │  |  ├── ...
│  │  │  ├── airport
│  │  │  │  ├── airport_00001.jpg
│  │  │  |  ├── airport_00002.jpg
│  │  │  |  ├── ...
|  |  ├── ...
│  │  ├── labels
│  │  │  ├── data
│  │  │  │  ├── airplane.csv
│  │  │  │  ├── airport.csv
│  │  │  │  ├── ...
│  │  │  ├── MLRSNet.csv
```

### UCM
This dataset has two versions, one with labels and one with captions. The following instructions prepare both of them.

1. Create a folder named "UCM" inside ``benchmarks/``.
2. Download the file ``UCM_captions.zip`` from [here](https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA).
3. Extract the zip file and open its content.
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

### RSICD
This dataset has two versions, one with labels and one with captions. The following instructions prepare both of them.

1. Create a folder named "RSICD" inside ``benchmarks/``.
2. Download the file ``RSICD_optimal-master.zip`` from [here](https://github.com/201528014227051/RSICD_optimal).
3. Extract the zip file and open its content.
4. Copy the file ``dataset_rsicd.json`` into ``benchmarks/RSICD/labels/sentences/``.
5. Extract the file ``txtclasses_rsicd.rar`` and copy its content into ``benchmarks/RSICD/labels/classes/``.
6. Extract the file ``RSICD_images.zip`` and copy its content into ``benchmarks/RSICD/images/``.
7. Copy the file ``metadata/RSICD/RSICD.csv`` into ``benchmarks/RSICD/labels/``.

Final folder structure
```
| benchmarks
│  ├── RSICD
│  │  ├── images
│  │  │  ├── 00001.jpg
│  │  │  ├── 00002.jpg
│  │  │  ├── ...
│  │  ├── labels
│  │  │  ├── classes
│  │  │  │  ├── airport.txt
│  │  │  │  ├── bare_land.txt
│  │  │  │  ├── ...
│  │  │  ├── sentences
│  │  │  │  ├── dataset_rsicd.json
│  │  │  ├── RSICD.csv
```

### RSITMD
This dataset has two versions, one with labels and one with captions. The following instructions prepare both of them.

1. Create a folder named "RSITMD" inside ``benchmarks/``.
2. Download the file ``RSITMD.zip`` from [here](https://github.com/xiaoyuan1996/AMFMN).
3. Extract the zip file and open its content.
4. Copy the content of the folder ``images`` into ``benchmarks/RSITMD/images/``.
5. Copy the file ``dataset_RSITMD.json`` into ``benchmarks/RSITMD/labels/``.
6. Copy the file ``metadata/RSITMD/RSITMD.csv`` into ``benchmarks/RSITMD/labels/``.

Final folder structure
```
| benchmarks
│  ├── RSITMD
│  │  ├── images
│  │  │  ├── airport_10.jpg
│  │  │  ├── airport_11.jpg
│  │  │  ├── ...
│  │  ├── labels
│  │  │  ├── dataset_RSITMD.json
│  │  │  ├── RSITMD.csv
```

### SIDNEY
This dataset has only captions, so it is not used for classification tasks.

1. Create a folder named "SIDNEY" inside ``benchmarks/``.
2. Download the file ``Sydney_captions.zip`` from [here](https://huggingface.co/datasets/isaaccorley/Sydney-Captions/tree/main).
3. Extract the zip file and open its content.
5. Copy the contents of the folder ``images`` into ``benchmarks/SIDNEY/images/``.
6. Copy the file ``metadata/SIDNEY/SIDNEY.csv`` into ``benchmarks/SIDNEY/labels/``.

Final folder structure
```
| benchmarks
│  ├── SIDNEY
│  │  ├── images
│  │  │  ├── 1.tif
│  │  │  ├── 2.tif
│  │  │  ├── ...
│  │  ├── labels
│  │  │  ├── SIDNEY.csv
```

