<center> 
<h1><strong>A benchmark for the evaluation of vision language embeddings for remote sensing applications</strong></h1>
<em>
Alberto Frizzera, info@albertofrizzera.com<br>
Riccardo Ricci, riccardo.ricci-1@unitn.it
</em>
<br>
</center>

## Introduction
This project aims at developing a platform for benchmarking of vision-language models in the remote sensing scenario. 

### DISCLAIMER
There is no restriction on the model type. The only requirement is that the model obey to three things:
1. Provide a preprocess function to preprocess the images.
2. Have an "encode_image" function, which takes a list of images and produces a list of embeddings. The images are supposed to be already preprocessed. 
3. Have a preprocess_text function to preprocess the text.
4. Have an "encode_text" function, which takes a list of texts and produces a list of embeddings. The texts are supposed to be already preprocessed.

## Installation

1. Create a conda environment following the instructions contained in ```environment.txt``` or using ```requirements.txt```.
2. Adjust the environmental variables of the dataset in ```.env``` in order to properly locate the datasets.

> **_Note:_**  The processed datasets will be available soon.

## Usage
1. Run the ```test.py``` by specifying the test parameters.
2. Collect the results in the ```reports/``` folder saved in a Latex document.

## Benchmark Datasets

The following list provides the datasets used to benchmark your model.

### Zero shot classification
- [X] [UCM](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
- [X] [WHU_RS19](https://captain-whu.github.io/BED4RS/#)
- [X] [RSSCN7](https://github.com/palewithout/RSSCN7)
- [X] [SIRI_WHU](http://www.lmars.whu.edu.cn/prof_web/zhongyanfei/e-code.html)
- [X] [RESISC45](https://figshare.com/articles/dataset/NWPU-RESISC45_Dataset_with_12_classes/16674166)
- [X] [RSI_CB128](https://github.com/lehaifeng/RSI-CB)
- [X] [RSI_CB256](https://github.com/lehaifeng/RSI-CB)
- [X] [EuroSAT](https://github.com/phelber/eurosat)
- [X] [PatternNet](https://sites.google.com/view/zhouwx/dataset)
- [X] [OPTIMAL_31](https://huggingface.co/datasets/jonathan-roberts1/Optimal-31)
- [X] [MLRSNet](https://github.com/cugbrs/MLRSNet)
- [X] [RSICD](https://github.com/201528014227051/RSICD_optimal)
- [X] [RSITMD](https://github.com/xiaoyuan1996/AMFMN)


### Image retrieval
- [X] [RSICD](https://github.com/201528014227051/RSICD_optimal)
- [X] [RSITMD](https://github.com/xiaoyuan1996/AMFMN)
- [X] [SIDNEY](https://mega.nz/folder/pG4yTYYA#4c4buNFLibryZnlujsrwEQ)
- [X] [UCM](https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA)

Datasets marked with [X] are already implemented and ready to use.

We are constantly updating the number of datasets that we support for testing. 
If needed, an exhaustive list of other satellite datasets is available [here](https://captain-whu.github.io/DiRS/).

To visualize the samples of all the above datasets, a web tool has been implemented (```web_app/main.py```)

## BASELINES
The following table report some baselines of CLIP-like models. Some are original, while others are finetuned for the remote sensing scenario.

### CLIP ViT-B/32

| Dataset         | Zero-shot Accuracy  | Linear-probe Accuracy | Recall@K (T2I)          | Recall@K (I2T)          |
|                 | (%)                 | (%)                  | R@1 / R@5 / R@10 / R@50 | R@1 / R@5 / R@10 / R@50 |

| UCM             | 64.52               | 95.0                 | 8.57 / 36.67 / 60.57 / 94.76  | 10.95 / 29.52 / 52.38 / 88.57 |
| WHU_RS19        | 80.6                | 99.0                 | -                         | -                         |
| RSSCN7          | 65.18               | 94.64                | -                         | -                         |
| SIRI_WHU        | 51.46               | 93.96                | -                         | -                         |
| RESISC45        | 75.52               | 97.95                | -                         | -                         |
| RSI_CB128       | 24.37               | 98.19                | -                         | -                         |
| RSI_CB256       | 34.02               | 98.93                | -                         | -                         |
| EuroSAT         | 39.83               | 95.3                 | -                         | -                         |
| PatternNet      | 58.83               | 98.82                | -                         | -                         |
| OPTIMAL_31      | 75.0                | 93.82                | -                         | -                         |
| MLRSNet         | 51.23               | 93.77                | -                         | -                         |
| RSICD           | 59.29               | 94.69                | 5.86 / 16.89 / 28.36 / 67.48  | 4.67 / 14.18 / 23.60 / 53.06 |
| RSITMD          | 53.54               | 93.14                | 8.72 / 27.79 / 42.57 / 77.57  | 9.51 / 23.01 / 34.07 / 62.39 |
| SIDNEY          | -                   | -                    | 12.07 / 41.38 / 68.97 / 100.00 | 12.07 / 39.66 / 55.17 / 86.21 |





<!-- <center> 
<img src="assets/report_benchmark.png" width="600"/>
</center> -->

## Dataset preparation
Each dataset should be downloaded and preprocessed. For each dataset, we delineate below the steps to accomplish to prepare it for the benchmarking.
First, create a folder named "benchmarks", wherever you want, and put its path in the .env file.

### UCM

### WHU_RS19

### RSSCN7

Steps:
1. Navigate to the "benchmarks" folder.
2. Clone the repository 
```bash
git clone https://github.com/palewithout/RSSCN7
```
3. Copy the file "metadata/RSSCN7/RSSCN7.pkl" inside "benchmarks/RSSCN7".

This dataset does not provide train-test-val splits in literature. We created random train-test-val splits using stratification, to ensure that the classes are balanced in each split.

### SIRI_WHU

### RESISC45

### RSI_CB128

### RSI_CB256

### EuroSAT

### PatternNet

### OPTIMAL_31

### MLRSNet

### RSICD

### RSITMD

### SIDNEY

