
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from dotenv import load_dotenv
import clip

# DATASETS FOR IMAGE CLASSIFICATION ONLY 

class EuroSAT(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for EuroSAT are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"EuroSAT/labels/EuroSAT.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"EuroSAT/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class OPTIMAL_31(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for OPTIMAL_31 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"OPTIMAL_31/labels/OPTIMAL_31.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"OPTIMAL_31/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    

class PatternNet(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for PatternNet are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/labels/PatternNet.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class MLRSNet(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for MLRSNet are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/labels/MLRSNet.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class WHU_RS19(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for WHU_RS19 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/labels/WHU_RS19.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)

        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class SIRI_WHU(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for SIRI_WHU are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/labels/SIRI_WHU.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class RSSCN7(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RSI_CB128 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSSCN7/labels/RSSCN7.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSSCN7/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class RSI_CB128(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RSI_CB128 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB128/labels/RSI_CB128.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB128/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class RSI_CB256(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RSI_CB256 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/labels/RSI_CB256.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class RESISC45(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RESISC45 are not available."
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/labels/RESISC45.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/images/",sample["filepath"])).convert("RGB")
        image = self.preprocess(image)
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
# DATASETs FOR IMAGE RETRIEVAL AND CLASSIFICATION
class UCM(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        if label_type=="label":
            data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM.pkl"),"rb"))
            self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        else:
            data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM_captions.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)

        self.n_samples = self.data.shape[0]
        self.preprocess = preprocess
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        if self.label_type=="label":
            image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/images/labels",sample["filepath"])).convert("RGB")
            target = sample[self.label_type]
        else:
            image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/images/captions",sample["filename"])).convert("RGB")
            target = clip.tokenize(sample[self.label_type], truncate=True)
        
        image = self.preprocess(image)
        
        return image, target

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class RSITMD(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/labels/RSITMD.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/images/",sample["filename"])).convert("RGB")
        image = self.preprocess(image)
        
        if self.label_type=="label":
            target = sample[self.label_type]
        else: 
            target = clip.tokenize(sample[self.label_type], truncate=True)
        
        return image, target

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class RSICD(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/labels/RSICD.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.unique_labels = np.sort(self.data["label"].unique()).tolist()
        self.preprocess = preprocess
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/images/",sample["filename"])).convert("RGB")
        image = self.preprocess(image)
        
        if self.label_type=="label":
            target = sample[self.label_type]
        else: 
            target = clip.tokenize(sample[self.label_type], truncate=True)
        
        return image, target

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class SIDNEY(Dataset):
    def __init__(self, split:str="test", label_type:str="label", preprocess:callable=None):
        assert label_type=="sentence", "Error! Class labels for SIDNEY are not available."
        load_dotenv()
        data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIDNEY/labels/SIDNEY.pkl"),"rb"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            
        self.n_samples = self.data.shape[0]
        self.preprocess = preprocess
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIDNEY/images/",sample["filename"])).convert("RGB")
        image = self.preprocess(image)
        
        target = clip.tokenize(sample[self.label_type], truncate=True)
        
        return image, target

    def __len__(self):
        return self.n_samples
    

def custom_collate(batch):
    images = []
    labels = []
    for sample in batch:
        images.append(sample["image"])
        labels.append(sample["label"])
        
    return images, labels


