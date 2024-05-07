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

### ORIGINAL CLIP MODELS FROM OPENAI
This tables reports the performance of the original CLIP models trained by OPENAI. You can find these models in their [github repository](https://github.com/openai/CLIP). 
All the zero-shot results are obtained using the following template "a remote sensing image of a {class}", where class is replaced by the class name.
### CLIP ViT-B/32
| Dataset    | Zero-shot Accuracy (%) | Linear-probe Accuracy (%) | Recall@K (T2I: R@1 / R@5 / R@10 / R@50) | Recall@K (I2T: R@1 / R@5 / R@10 / R@50) |
|------------|------------------------|---------------------------|----------------------------------------|----------------------------------------|
| UCM        | 64.52                  | 95.0                      | 8.57 / 36.67 / 60.57 / 94.76           | 10.95 / 29.52 / 52.38 / 88.57          |
| WHU_RS19   | 80.6                   | 99.0                      | -                                      | -                                      |
| RSSCN7     | 65.18                  | 94.64                     | -                                      | -                                      |
| SIRI_WHU   | 51.46                  | 93.96                     | -                                      | -                                      |
| RESISC45   | 75.52                  | 97.95                     | -                                      | -                                      |
| RSI_CB128  | 24.37                  | 98.19                     | -                                      | -                                      |
| RSI_CB256  | 34.02                  | 98.93                     | -                                      | -                                      |
| EuroSAT    | 39.83                  | 95.3                      | -                                      | -                                      |
| PatternNet | 58.83                  | 98.82                     | -                                      | -                                      |
| OPTIMAL_31 | 75.0                   | 93.82                     | -                                      | -                                      |
| MLRSNet    | 51.23                  | 93.77                     | -                                      | -                                      |
| RSICD      | 59.29                  | 94.69                     | 5.86 / 16.89 / 28.36 / 67.48           | 4.67 / 14.18 / 23.60 / 53.06          |
| RSITMD     | 53.54                  | 93.14                     | 8.72 / 27.79 / 42.57 / 77.57           | 9.51 / 23.01 / 34.07 / 62.39          |
| SIDNEY     | -                      | -                         | 12.07 / 41.38 / 68.97 / 100.00         | 12.07 / 39.66 / 55.17 / 86.21         |

### CLIP ViT-B/16
| Dataset    | Zero-shot Accuracy (%) | Linear-probe Accuracy (%) | Recall@K (T2I: R@1 / R@5 / R@10 / R@50) | Recall@K (I2T: R@1 / R@5 / R@10 / R@50) |
|------------|------------------------|---------------------------|----------------------------------------|----------------------------------------|
| UCM        | 70.24                  | 96.67                     | 9.81 / 39.14 / 66.57 / 94.76           | 8.10 / 38.10 / 61.43 / 93.33          |
| WHU_RS19   | 81.09                  | 99.0                      | -                                      | -                                      |
| RSSCN7     | 65.89                  | 93.04                     | -                                      | -                                      |
| SIRI_WHU   | 50.21                  | 93.75                     | -                                      | -                                      |
| RESISC45   | 72.19                  | 98.43                     | -                                      | -                                      |
| RSI_CB128  | 26.01                  | 98.28                     | -                                      | -                                      |
| RSI_CB256  | 37.9                   | 98.95                     | -                                      | -                                      |
| EuroSAT    | 44.91                  | 95.61                     | -                                      | -                                      |
| PatternNet | 64.49                  | 98.8                      | -                                      | -                                      |
| OPTIMAL_31 | 73.39                  | 93.82                     | -                                      | -                                      |
| MLRSNet    | 53.73                  | 94.58                     | -                                      | -                                      |
| RSICD      | 60.2                   | 96.25                     | 5.89 / 19.18 / 29.19 / 67.76           | 6.50 / 18.21 / 27.36 / 55.54          |
| RSITMD     | 55.75                  | 95.8                      | 8.14 / 29.12 / 46.02 / 80.00           | 11.28 / 25.00 / 33.63 / 61.95         |
| SIDNEY     | -                      | -                         | 11.03 / 47.93 / 69.31 / 100.00         | 8.62 / 27.59 / 53.45 / 91.38          |

### CLIP ViT-L/14
| Dataset    | Zero-shot Accuracy (%) | Linear-probe Accuracy (%) | Recall@K (T2I: R@1 / R@5 / R@10 / R@50) | Recall@K (I2T: R@1 / R@5 / R@10 / R@50) |
|------------|------------------------|---------------------------|----------------------------------------|----------------------------------------|
| UCM        | 73.1                   | 97.86                     | 10.67 / 45.71 / 73.05 / 97.62          | 9.52 / 38.57 / 65.24 / 93.33           |
| WHU_RS19   | 82.59                  | 100.0                     | -                                      | -                                      |
| RSSCN7     | 56.25                  | 95.18                     | -                                      | -                                      |
| SIRI_WHU   | 55.62                  | 95.21                     | -                                      | -                                      |
| RESISC45   | 83.14                  | 98.86                     | -                                      | -                                      |
| RSI_CB128  | 34.15                  | 98.65                     | -                                      | -                                      |
| RSI_CB256  | 46.26                  | 99.49                     | -                                      | -                                      |
| EuroSAT    | 48.98                  | 97.19                     | -                                      | -                                      |
| PatternNet | 71.07                  | 99.29                     | -                                      | -                                      |
| OPTIMAL_31 | 79.84                  | 97.31                     | -                                      | -                                      |
| MLRSNet    | 60.12                  | 95.99                     | -                                      | -                                      |
| RSICD      | 61.02                  | 96.25                     | 5.03 / 19.07 / 30.21 / 69.39           | 6.31 / 17.38 / 27.54 / 58.28           |
| RSITMD     | 61.5                   | 98.01                     | 11.42 / 32.92 / 47.35 / 79.78          | 10.62 / 28.10 / 39.16 / 67.70          |
| SIDNEY     | -                      | -                         | 14.83 / 47.59 / 70.69 / 100.00         | 13.79 / 39.66 / 51.72 / 87.93          |

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

