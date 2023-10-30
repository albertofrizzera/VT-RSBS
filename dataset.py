# -------------------------------------------------------------------------
# Author:   Alberto Frizzera, info@albertofrizzera.com
# Date:     28/08/2023
# -------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def custom_collate(batch):
    X = dict()
    for key in batch[0][0].keys():
        X[key] = [item[0][key] for item in batch]
    batch = [X,
            [item[1] for item in batch]]
    return batch


class BigEarthNet(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=12, crop=False, text_template="", encoder_type=None, label_type="sentence", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        # Note! This dataset contains multiple labels per image, so the variable n_sents indicates how many labels per image will be taken.
        assert label_type=="label", "Error! Sentences for BigEarthNet are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"BigEarthNet/labels/BigEarthNet.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        # Select the first N random labels for each image
        assert n_sents>=1 and n_sents<=12, "Error! Select from 0 to maximum 5 sentences for each sample."
        self.data = self.data.sample(frac=1).groupby('filename').head(n_sents).reset_index(drop=True)
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def normalize(self, band):
        band_min, band_max = (band.min(), band.max())
        return ((band-band_min)/((band_max - band_min)))
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        filename = sample["filename"]
        
        band_2 = np.array(Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"BigEarthNet/images/",filename,filename+"_B02.tif")))
        band_2 = np.round(self.normalize(band_2)*255).astype('uint8')

        band_3 = np.array(Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"BigEarthNet/images/",filename,filename+"_B03.tif")))
        band_3 = np.round(self.normalize(band_3)*255).astype('uint8')

        band_4 = np.array(Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"BigEarthNet/images/",filename,filename+"_B04.tif")))
        band_4 = np.round(self.normalize(band_4)*255).astype('uint8')

        image = Image.fromarray(np.moveaxis(np.stack([band_4, band_3, band_2]),0,-1))
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples


class EuroSAT(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for EuroSAT are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"EuroSAT/labels/EuroSAT.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":256} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"EuroSAT/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples

    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"EuroSAT/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class MLRSNet(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for MLRSNet are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/labels/MLRSNet.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples

    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class NWPU(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=1, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"NWPU/labels/NWPU.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        if label_type=="label":
            self.data = self.data.drop_duplicates(subset=["filepath"], keep='first').reset_index(drop=True)
        else:
            assert text_template=="", "Error! With sentences the text template must be empty."
            # Select the first N random sentences for each image
            assert n_sents>=1 and n_sents<=5, "Error! Select from 0 to maximum 5 sentences for each sample."
            self.data = self.data.sample(frac=1).groupby('filepath').head(n_sents).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.label_type = label_type
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"NWPU/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample[self.label_type]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples

    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"NWPU/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class OPTIMAL_31(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for OPTIMAL_31 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"OPTIMAL_31/labels/OPTIMAL_31.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"OPTIMAL_31/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples

    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"OPTIMAL_31/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class PatternNet(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for PatternNet are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/labels/PatternNet.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples

    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class RESISC45(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for RESISC45 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/labels/RESISC45.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples
    
    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class RSI_CB256(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for RSI_CB256 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/labels/RSI_CB256.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples
    
    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class RSICD(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=1, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/labels/RSICD.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        if label_type=="label":
            self.data = self.data.drop_duplicates(subset=["filename"], keep='first').reset_index(drop=True)
        else:
            assert text_template=="", "Error! With sentences the text template must be empty."
            # Select the first N random sentences for each image
            assert n_sents>=1 and n_sents<=5, "Error! Select from 0 to maximum 5 sentences for each sample."
            self.data = self.data.sample(frac=1).groupby('filename').head(n_sents).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.label_type = label_type
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512, "image_retrieval":128} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/images/",sample["filename"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample[self.label_type]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples
    
    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/images/",sample["filename"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class RSITMD(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=1, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/labels/RSITMD.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        if label_type=="label":
            self.data = self.data.drop_duplicates(subset=["filename"], keep='first').reset_index(drop=True)
        else:
            assert text_template=="", "Error! With sentences the text template must be empty."
            # Select the first N random sentences for each image
            assert n_sents>=1 and n_sents<=5, "Error! Select from 0 to maximum 5 sentences for each sample."
            self.data = self.data.sample(frac=1).groupby('filename').head(n_sents).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.label_type = label_type
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512, "image_retrieval":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/images/",sample["filename"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample[self.label_type]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples
    
    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/images/",sample["filename"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class SIDNEY(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=1, crop=False, text_template="", encoder_type=None, label_type="sentence", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="sentence", "Error! Class labels for SIDNEY are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIDNEY/labels/SIDNEY.pkl"),"rb"))
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        assert text_template=="", "Error! With sentences the text template must be empty."
        # Select the first N random sentences for each image
        assert n_sents>=1 and n_sents<=5, "Error! Select from 0 to maximum 5 sentences for each sample."
        self.data = self.data.sample(frac=1).groupby('filename').head(n_sents).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        self.preprocess = preprocess
        self.text_template = text_template
        self.label_type = label_type
        self.n_samples = self.data.shape[0]
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"image_retrieval":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIDNEY/images/",sample["filename"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["sentence"]
        
        return {"image":X}, Y

    def __len__(self):
        return self.n_samples


class SIRI_WHU(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for SIRI_WHU are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/labels/SIRI_WHU.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples

    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class UCM(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=1, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        if label_type=="label":
            data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM.pkl"),"rb"))
        else:
            data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM_captions.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        if label_type=="sentence":
            assert text_template=="", "Error! With sentences the text template must be empty."
            # Select the first N random sentences for each image
            assert n_sents>=1 and n_sents<=5, "Error! Select from 0 to maximum 5 sentences for each sample."
            self.data = self.data.sample(frac=1).groupby('filename').head(n_sents).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.label_type = label_type
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        if label_type=="label":
            self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512, "image_retrieval":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        if self.label_type=="label":
            image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/images/labels",sample["filepath"])).convert("RGB")
        else:
            image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/images/captions",sample["filename"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample[self.label_type]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        if self.label_type=="label":
            return {"image":X, "label":sample["label"]}, Y
        return {"image":X}, Y

    def __len__(self):
        return self.n_samples
    
    # For plitting utilities
    def custom_label(self, label):
        assert self.label_type=="label", "Error! Label filtering can be performed only with label_type='label'."
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/images/labels",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y


class WHU_RS19(Dataset):

    def __init__(self, preprocess=None, split="train", split2=None, split3=None, n_sents=None, crop=False, text_template="", encoder_type=None, label_type="label", captioner_name=None, prompt_type=None, return_PIL=False, samples_per_label=None):
        assert label_type=="label", "Error! Sentences for WHU_RS19 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/labels/WHU_RS19.pkl"),"rb"))
        if encoder_type=="one_hot_encoder":
            assert text_template=="", "Error! With class encoders (e.g. in few-shot classification) the text template has to be empty."
            self.class_encoder = OneHotEncoder()
            self.class_encoder.fit(data_total[["label"]].values)
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            if split2:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split2]]).reset_index(drop=True)
            if split3:
                self.data = pd.concat([self.data, data_total[data_total["split"]==split3]]).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True) # Shuffle
        # Select N samples per label for few-shot classification
        if samples_per_label:
            self.data = self.data.groupby('label').head(samples_per_label).reset_index(drop=True)
        self.preprocess = preprocess
        self.text_template = text_template
        self.encoder_type = encoder_type
        self.n_samples = self.data.shape[0]
        self.unique_labels = self.text_template + np.sort(self.data["label"].unique())
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.return_PIL = return_PIL
        self.batch_size = {"zero_shot":512} # Default batch_size considering 24GB of GPU memory
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/images/",sample["filepath"])).convert("RGB")
        
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        
        # Encoding labels for classification, if encoder_type is specified, assuming text_template=""
        if self.encoder_type=="one_hot_encoder":
            Y = self.class_encoder.transform([[Y]]).astype(int).toarray()
        
        return {"image":X, "label":sample["label"]}, Y

    def __len__(self):
        return self.n_samples

    # For plitting utilities
    def custom_label(self, label):
        sample = self.data[self.data["label"]==label].reset_index(drop=True).sample(n=1).iloc[0]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/images/",sample["filepath"])).convert("RGB")
        if self.return_PIL:
            X = image
        elif self.preprocess:
            X = self.preprocess(image)
        else:
            X = self.transform(image)
        Y = self.text_template + sample["label"]
        return X, Y