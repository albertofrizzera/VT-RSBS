
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import ast
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from dotenv import load_dotenv
import pandas as pd

# DATASETS FOR IMAGE CLASSIFICATION ONLY 

class EuroSAT(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for EuroSAT are not available."
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"EuroSAT/labels/EuroSAT.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"EuroSAT/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class OPTIMAL_31(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for OPTIMAL_31 are not available."
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"OPTIMAL_31/labels/OPTIMAL_31.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"OPTIMAL_31/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    

class PatternNet(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for PatternNet are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/labels/PatternNet.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/labels/PatternNet.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"PatternNet/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class MLRSNet(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for MLRSNet are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/labels/MLRSNet.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/labels/MLRSNet.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"MLRSNet/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class WHU_RS19(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for WHU_RS19 are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/labels/WHU_RS19.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/labels/WHU_RS19.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)

        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"WHU_RS19/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class SIRI_WHU(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for SIRI_WHU are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/labels/SIRI_WHU.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/labels/SIRI_WHU.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIRI_WHU/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class RSSCN7(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RSI_CB128 are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSSCN7/labels/RSSCN7.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSSCN7/labels/RSSCN7.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSSCN7/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class RSI_CB128(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RSI_CB128 are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB128/labels/RSI_CB128.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB128/labels/RSI_CB128.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB128/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class RSI_CB256(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RSI_CB256 are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/labels/RSI_CB256.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/labels/RSI_CB256.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSI_CB256/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class RESISC45(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        assert label_type=="label", "Error! Sentences for RESISC45 are not available."
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/labels/RESISC45.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/labels/RESISC45.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RESISC45/images/",sample["filepath"])).convert("RGB")
        
        return image, sample["label"]

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
# DATASETs FOR IMAGE RETRIEVAL AND CLASSIFICATION
class UCM(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        if label_type=="label":
            #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM.pkl"),"rb"))
            data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM.csv"))
            self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        else:
            #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM_captions.pkl"),"rb"))
            data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/labels/UCM_captions.csv"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)

        self.n_samples = self.data.shape[0]
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        if self.label_type=="label":
            image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/images/labels",sample["filepath"])).convert("RGB")
            target = sample[self.label_type]
        else:
            image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"UCM/images/captions",sample["filename"])).convert("RGB")
            # Get target (it is a list in str format, using ast to convert it to a list)
            target = sample[self.label_type]
            target = ast.literal_eval(target)
            target = [n.strip() for n in target]
        
        return image, target

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class RSITMD(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/labels/RSITMD.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/labels/RSITMD.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSITMD/images/",sample["filename"])).convert("RGB")
        # Get target (it is a list in str format, using ast to convert it to a list)
        target = sample[self.label_type]
        target = ast.literal_eval(target)
        target = [n.strip() for n in target]
        
        return image, target

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels

class RSICD(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        # Load dotenv to load the paths in the environment variables
        load_dotenv()
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/labels/RSICD.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/labels/RSICD.csv"))
        self.unique_labels = np.sort(data_total["label"].unique()).tolist()
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
        
        self.n_samples = self.data.shape[0]
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"RSICD/images/",sample["filename"])).convert("RGB")
        # Get target (it is a list in str format, using ast to convert it to a list)
        target = sample[self.label_type]
        target = ast.literal_eval(target)
        target = [n.strip() for n in target]

        return image, target

    def __len__(self):
        return self.n_samples
    
    def _get_unique_labels(self):
        return self.unique_labels
    
class SIDNEY(Dataset):
    def __init__(self, split:str="test", label_type:str="label"):
        assert label_type=="sentence", "Error! Class labels for SIDNEY are not available."
        load_dotenv()
        #data_total = pickle.load(open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIDNEY/labels/SIDNEY.pkl"),"rb"))
        data_total = pd.read_csv(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIDNEY/labels/SIDNEY.csv"))
        
        if split=="tot":
            self.data = data_total
        else:
            self.data = data_total[data_total["split"]==split].reset_index(drop=True)
            
        self.n_samples = self.data.shape[0]
        self.label_type = label_type
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(os.environ["BENCHMARK_DATASETS"],"SIDNEY/images/",sample["filename"])).convert("RGB")
        # Get target (it is a list in str format, using ast to convert it to a list)
        target = sample[self.label_type]
        target = ast.literal_eval(target)
        target = [n.strip() for n in target]
        
        return image, target

    def __len__(self):
        return self.n_samples
    
def custom_collate_fn(batch):
    images = []
    labels = []
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])

    return images, labels
