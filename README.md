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

## Disclaimer
There is no restriction on the model type. The only requirement is providing these three things:
1. A function to load the model. This function must return three callables: model, textprocessor, imageprocessor.
- The model is the model itself.
- Textprocessor is a function that takes in input a list of strings and tokenize them, producing in output a tensor of indices of size BxL (tensors padded at the maximum length). 
- Imageprocessor is a function that takes in input a single PIL Image and apply transformations on it, producing in output a tensor of size CxHxW.

2. A function to produce text embeddings. This function takes in input the model, the texts (list of strings), and the device. It then preprocess the text with textprocessor and produce embeddings with the model. It returns the embeddings, a tensor of shape BxD, where D is the embedding dimension.

3. A function to produce image embeddings. This function takes in input the model, the images (list of PIL.Image objects), and the device. It then preprocess the images with imageprocessor and produce embeddings with the model. It returns the embeddings, a tensor of shape BxD, where D is the embedding dimension.

You can find some examples of the implementation of these functions in ```utils.py```. Specifically, functions to load remoteCLIP, georsCLIP, CLIPrsicdv2 and openaiCLIP have been implemented. 

## Installation
1. Create a conda environment following the instructions contained in ```environment.txt``` or using ```requirements.txt```.
2. Adjust the environmental variables of the dataset in ```.env``` in order to properly locate the datasets.

> **_Note:_**  We are working to provide instructions to download and prepare all the datasets.

## Usage
1. Modify the beginning of the module ```eval.py``` by importing your custom functions, and replacing them after "load_function", "encode_text_fn" and "encode_image_fn".
2. Modify the templates, placing the ones that you want to use for evaluation.
3. Run ```eval.py```.
4. Collect the results in the ```reports/``` folder saved in a Latex document.

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

## Benchmark Models
The following figures report some baselines of CLIP-like models. Some are original, while others are finetuned for the remote sensing scenario.

For a detailed breakdown on each dataset, refer to the [report](reports/single_model_breakdown.md).

### Zero-shot classification
![alt text](assets/zero_shot_acc.png)
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
Steps:
1. Click on this link: [Patternet](https://nuisteducn1-my.sharepoint.com/:u:/g/personal/zhouwx_nuist_edu_cn/EYSPYqBztbBBqS27B7uM_mEB3R9maNJze8M1Qg9Q6cnPBQ?e=MSf977)
2. Download the file "PatternNet.zip".
3. Unzip the file and copy the folder "PatterNet" inside "benchmarks". The folder structure inside "PatterNet" should be this 
```
PatternNet
│
└───images
│   │   airplane
│   │   baseball_field
│   │   ...

### OPTIMAL_31

### MLRSNet

### RSICD

### RSITMD

### SIDNEY

