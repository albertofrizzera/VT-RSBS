# -------------------------------------------------------------------------
# Author:   Alberto Frizzera, info@albertofrizzera.com
# Date:     11/10/2023
# -------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
import pandas as pd
import clip
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
import warnings
import multiprocessing
import time
import json
import shutil
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from dataset.dataset import *
from utils import time_convert, build_report, parse_line_args


if __name__ == '__main__':
    
    load_dotenv()
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', 
                        default="True",
                        help="Logger")
    parser.add_argument('--RUN_ID',
                        required=False,
                        default=None)
    parser.add_argument('--RUN_epoch', 
                        type=int,
                        required=False,
                        default=None)
    args = parser.parse_args()
    assert args.log=="True" or args.log=="False", "Error! Logger choice not valid."
    
    start_time = time.time()
    initial_datetime = datetime.now().strftime("%Y%m%d_%H.%M.%S")
    # initial_datetime = "CLIP_baseline"
    # initial_datetime = "RemoteCLIP_baseline"
    
    params = {"model": "CLIP",
              "model_checkpoints": "CLIP",
                                    # "RemoteCLIP",
              "datasets": {"zero_shot": ["EuroSAT","MLRSNet","OPTIMAL_31","PatternNet","RESISC45","RSICD","RSITMD","SIRI_WHU","UCM","WHU_RS19"], # "BigEarthNet","EuroSAT","MLRSNet","OPTIMAL_31","PatternNet","RESISC45","RSI_CB256","RSICD","RSITMD","SatCLIP","SIRI_WHU","UCM","WHU_RS19"
                           "image_retrieval": ["RSICD","RSITMD","SIDNEY","UCM"]}, # "RSICD","RSITMD","SIDNEY","UCM" - Too big datasets: "NWPU","SatCLIP"
              "text_template": "a satellite photo of a ", # "a remote sensing image of a ", "an image of a ", "a satellite image of a ", "a satellite photo of a "
              "lnr_prob_max_iterations": 1000,
              "n_neighbors": 10,
              "image_retrieval_N_rank": 20,
              "include_baseline": True,
              }
    params = parse_line_args(args, params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loading model -----------------------------------------------------------------------------------------------------
    clip_model, preprocess = clip.load("ViT-B/32", download_root=os.path.join(os.path.dirname(__file__), "saved_models"))
    # Restoring checkpoints
    # --------------- LOAD YOUR CHECKPOINTS -----------------
    # clip_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "saved_models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"), map_location=device))
    clip_model.to(device).eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    collate_fn = torch.utils.data.default_collate
    # -------------------------------------------------------------------------------------------------------------------
    
    # Datasets
    print("Loading datasets: Started...", end="\r")
    datasets = {"zero_shot":{"train":[], "val":[], "test":[], "tot":[]},
                "lnr_prob":{"train":[], "val":[], "test":[], "tot":[]},
                "image_retrieval":{"train":[], "val":[], "test":[], "tot":[]}}
    for split in ["train","val","test","tot"]:
        for dataset_name in params["datasets"]["zero_shot"]:
            datasets["zero_shot"][split].append(globals()[dataset_name](preprocess, split, crop=True, text_template=params["text_template"], label_type="label"))
        for dataset_name in params["datasets"]["zero_shot"]:
            if dataset_name=="SatCLIP":
                datasets["lnr_prob"][split].append(globals()[dataset_name](preprocess, split, crop=True, label_type="label", encoder_type="one_hot_encoder", samples_per_label=1000))
            else:
                datasets["lnr_prob"][split].append(globals()[dataset_name](preprocess, split, crop=True, label_type="label", encoder_type="one_hot_encoder"))
        for dataset_name in params["datasets"]["image_retrieval"]:
            datasets["image_retrieval"][split].append(globals()[dataset_name](preprocess, split, crop=True, label_type="sentence", n_sents=1))
    print("Loading datasets: Done      ")
    
    # Dataloaders
    print("Loading dataloaders: Started...", end="\r")
    dataloaders = {"zero_shot":{"train":[], "val":[], "test":[], "tot":[]},
                    "lnr_prob":{"train":[], "val":[], "test":[], "tot":[]},
                    "image_retrieval":{"train":[], "val":[], "test":[], "tot":[]}}
    for split in ["train","val","test","tot"]:
        for dataset in datasets["zero_shot"][split]:
            dataloaders["zero_shot"][split].append(DataLoader(dataset=dataset, batch_size=dataset.batch_size["zero_shot"], shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.Pool()._processes-4))
        for dataset in datasets["lnr_prob"][split]:
            if split=="train":
                dataloaders["lnr_prob"][split].append(DataLoader(dataset=dataset, batch_size=dataset.batch_size["zero_shot"], shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.Pool()._processes-4))
            else:
                dataloaders["lnr_prob"][split].append(DataLoader(dataset=dataset, batch_size=dataset.batch_size["zero_shot"], shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.Pool()._processes-4))
        for dataset in datasets["image_retrieval"][split]:
            dataloaders["image_retrieval"][split].append(DataLoader(dataset=dataset, batch_size=dataset.batch_size["image_retrieval"], shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.Pool()._processes-4))
    print("Loading dataloaders: Done      ")
    
    # Log DataFrame
    report_log = pd.DataFrame(columns=["dataset_name","dataset_type","accuracy"])
    
    # Zero-shot classification
    print("Zero-shot classification: Started...")
    for dataset, dataloader in zip(datasets["zero_shot"]["test"], dataloaders["zero_shot"]["test"]):
        samples = 0
        cumulative_corrects = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12))
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images.to(device))
                text_tokens = clip.tokenize(dataset.unique_labels).to(device)
                text_features = clip_model.encode_text(text_tokens)
                
                text_features = torch.div(text_features, text_features.norm(dim=1, keepdim=True))
                image_features = torch.div(image_features, image_features.norm(dim=1, keepdim=True))
                similarity_matrix = torch.matmul(text_features, image_features.t())
                pred = torch.argmax(similarity_matrix, dim=0).cpu().numpy()
                truth = np.searchsorted(dataset.unique_labels, Y)
                
                cumulative_corrects += sum(pred == truth)
                samples += len(X)
                cumulative_accuracy = cumulative_corrects / samples
                tepoch.set_postfix(score=cumulative_accuracy)
        report_log = pd.concat([report_log, pd.DataFrame([{"dataset_name":dataset.__class__.__name__, "dataset_type":"zero_shot", "accuracy":cumulative_accuracy}])]).reset_index(drop=True)
    print("Zero-shot classification: Done      ")
    
    # Linear probing
    print("Linear probing: Started...")
    # Iterating each dataset
    for i in range(len(datasets["lnr_prob"]["train"])):
        
        # Train
        dataset = datasets["lnr_prob"]["train"][i]
        dataloader = dataloaders["lnr_prob"]["train"][i]
        df_train = pd.DataFrame()
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12)+" - "+"Train")
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images)
                # Target class
                classes = Y.squeeze()
                temp_df = pd.DataFrame(image_features.cpu().numpy())
                temp_df["class"] = pd.Series(classes.argmax(dim=1))
                df_train = pd.concat([df_train, temp_df]).reset_index(drop=True)
        
        # Test
        dataset = datasets["lnr_prob"]["test"][i]
        dataloader = dataloaders["lnr_prob"]["test"][i]
        df_test = pd.DataFrame()
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12)+" - "+"Test ")
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images)
                # Target class
                classes = Y.squeeze()
                temp_df = pd.DataFrame(image_features.cpu().numpy())
                temp_df["class"] = pd.Series(classes.argmax(dim=1))
                df_test = pd.concat([df_test, temp_df]).reset_index(drop=True)
        
        X_train = df_train.drop("class", axis=1).to_numpy()
        X_test = df_test.drop("class", axis=1).to_numpy()
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=params["lnr_prob_max_iterations"], verbose=False)
        classifier.fit(X_train, df_train["class"])
        y_pred = classifier.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(df_test["class"], y_pred)
        print(dataset.__class__.__name__.ljust(12)+" - "+"Accuracy: ",np.round(accuracy,3))
        report_log = pd.concat([report_log, pd.DataFrame([{"dataset_name":dataset.__class__.__name__, "dataset_type":"lnr_prob", "accuracy":accuracy}])]).reset_index(drop=True)
    print("Linear probing: Done      ")
    
    # KNN classification
    print("KNN classification: Started...")
    # Iterating each dataset
    for i in range(len(datasets["lnr_prob"]["train"])):
        
        # Train
        dataset = datasets["lnr_prob"]["train"][i]
        dataloader = dataloaders["lnr_prob"]["train"][i]
        df_train = pd.DataFrame()
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12)+" - "+"Train")
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images)
                # Target class
                classes = Y.squeeze()
                temp_df = pd.DataFrame(image_features.cpu().numpy())
                temp_df["class"] = pd.Series(classes.argmax(dim=1))
                df_train = pd.concat([df_train, temp_df]).reset_index(drop=True)
        
        # Test
        dataset = datasets["lnr_prob"]["test"][i]
        dataloader = dataloaders["lnr_prob"]["test"][i]
        df_test = pd.DataFrame()
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12)+" - "+"Test ")
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images)
                # Target class
                classes = Y.squeeze()
                temp_df = pd.DataFrame(image_features.cpu().numpy())
                temp_df["class"] = pd.Series(classes.argmax(dim=1))
                df_test = pd.concat([df_test, temp_df]).reset_index(drop=True)
        
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(df_train.drop("class", axis=1))
        X_test = scaler.transform(df_test.drop("class", axis=1))
        knn = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
        knn.fit(X_train, df_train["class"])
        y_pred = knn.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(df_test["class"], y_pred)
        print(dataset.__class__.__name__.ljust(12)+" - "+"Accuracy: ",np.round(accuracy,3))
        report_log = pd.concat([report_log, pd.DataFrame([{"dataset_name":dataset.__class__.__name__, "dataset_type":"knn", "accuracy":accuracy}])]).reset_index(drop=True)
    print("KNN classification: Done      ")
    
    # Image2Text
    print("Image2Text: Started...")
    for dataset, dataloader in zip(datasets["image_retrieval"]["tot"], dataloaders["image_retrieval"]["tot"]):
        
        text_tokens = clip.tokenize(dataset.data["sentence"].unique()).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = torch.div(text_features, text_features.norm(dim=1, keepdim=True))
        
        samples = 0
        cumulative_corrects = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12))
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images)
                image_features = torch.div(image_features, image_features.norm(dim=1, keepdim=True))
                similarity_matrix = torch.matmul(text_features, image_features.t())
                
                for i in range(X.shape[0]):
                    scores_df = pd.DataFrame(similarity_matrix.cpu()[:,i])
                    scores_df["sentence"] = dataset.data["sentence"].unique()
                    scores_df = scores_df.sort_values(by=[0], ascending=False).reset_index(drop=True)
                    texts_per_image = dataset.data_all_texts[dataset.data_all_texts["image_id"]==inputs["image_id"][i]]["sentence"].values
                    if np.any(np.isin(scores_df.iloc[:params["image_retrieval_N_rank"]]["sentence"].values, texts_per_image)):
                        cumulative_corrects += 1
                
                samples += len(X)
                cumulative_accuracy = cumulative_corrects / samples
                tepoch.set_postfix(score=cumulative_accuracy)
        report_log = pd.concat([report_log, pd.DataFrame([{"dataset_name":dataset.__class__.__name__, "dataset_type":"image2text", "accuracy":cumulative_accuracy}])]).reset_index(drop=True)
    print("Image2Text: Done      ")
    
    # Text2Image
    print("Text2Image: Started...")
    for dataset, dataloader in zip(datasets["image_retrieval"]["tot"], dataloaders["image_retrieval"]["tot"]):
        
        text_tokens = clip.tokenize(dataset.data["sentence"].unique()).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = torch.div(text_features, text_features.norm(dim=1, keepdim=True))
        
        samples = 0
        cumulative_corrects = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12)+" - Step 1/2")
                
                # Extracting clip features
                images = X.to(device)
                if samples==0:
                    image_features = clip_model.encode_image(images)
                    image_ids = np.array(inputs["image_id"])
                else:
                    image_features = torch.concat((image_features,clip_model.encode_image(images)))
                    image_ids = np.concatenate((image_ids,inputs["image_id"]))
                samples += len(X)
        
        image_features = torch.div(image_features, image_features.norm(dim=1, keepdim=True))
        similarity_matrix = torch.matmul(text_features, image_features.t())
        
        for i in tqdm(range(len(dataset.data["sentence"].unique())), desc=dataset.__class__.__name__.ljust(12)+" - Step 2/2"):
            scores_df = pd.DataFrame(similarity_matrix.cpu()[i,:])
            scores_df["image_id"] = image_ids
            scores_df = scores_df.sort_values(by=[0], ascending=False).reset_index(drop=True)
            images_per_text = dataset.data_all_texts[dataset.data_all_texts["sentence"]==dataset.data["sentence"].unique()[i]]["image_id"].values
            if np.any(np.isin(scores_df.iloc[:params["image_retrieval_N_rank"]]["image_id"].values, images_per_text)):
                cumulative_corrects += 1
        
        accuracy = cumulative_corrects / samples
        print(dataset.__class__.__name__.ljust(12)+" - "+"Accuracy: ",np.round(accuracy,3))
        report_log = pd.concat([report_log, pd.DataFrame([{"dataset_name":dataset.__class__.__name__, "dataset_type":"text2image", "accuracy":accuracy}])]).reset_index(drop=True)
    print("Text2Image: Done      ")
    
    # Save report log
    if args.log=="True":
        if os.path.exists(os.path.join(os.path.dirname(__file__),"reports",initial_datetime)):
            shutil.rmtree(os.path.join(os.path.dirname(__file__),"reports",initial_datetime))
        os.makedirs(os.path.join(os.path.dirname(__file__),"reports",initial_datetime))
        report_log.reset_index(drop=True, inplace=True)
        report_log.to_csv(os.path.join(os.path.dirname(__file__),"reports",initial_datetime,"report_"+initial_datetime+".csv"), index=True, lineterminator='\r\n')
        json.dump(params, open(os.path.join(os.path.dirname(__file__),"reports",initial_datetime,"report_"+initial_datetime+".json"), "w"))
        build_report(params, report_log, initial_datetime, include_baseline=params["include_baseline"])
    print("=========================================")
    print(report_log)
    print("=========================================")
    
    execution_time = int(time.time() - start_time)
    print("Execution time: ",time_convert(execution_time))