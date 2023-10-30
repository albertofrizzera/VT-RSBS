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
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
import neptune
from neptune.types import File
import warnings
import multiprocessing
import time
import json
import shutil
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

from dataset import *
from utils import time_convert, build_report


if __name__ == '__main__':
    
    load_dotenv()
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', 
                        default="True",
                        help="Logger")
    args = parser.parse_args()
    assert args.log=="True" or args.log=="False", "Error! Logger choice not valid."
    
    start_time = time.time()
    initial_datetime = datetime.now().strftime("%Y%m%d_%H.%M.%S")
    # initial_datetime = "CLIP_baseline"
    
    params = {"model": "CLIP",
              "model_checkpoints": "...folder/checkpoints.pth",
              "datasets": {"zero_shot": ["BigEarthNet","EuroSAT","MLRSNet","OPTIMAL_31","PatternNet","RESISC45","RSI_CB128","RSI_CB256","RSICD","RSITMD","SatCLIP","SIRI_WHU","UCM","WHU_RS19"], # "BigEarthNet","EuroSAT","MLRSNet","OPTIMAL_31","PatternNet","RESISC45","RSI_CB128","RSI_CB256","RSICD","RSITMD","SatCLIP","SIRI_WHU","UCM","WHU_RS19"
                           "image_retrieval": ["RSICD","RSITMD","SIDNEY","UCM"]}, # "RSICD","RSITMD","SIDNEY","UCM" - Too big datasets: "NWPU","SatCLIP"
              "text_template": "a remote sensing image of a ", # "a remote sensing image of a ", "an image of a ", "a satellite image of a "
              "few_shot_loss": "BCEWithLogitsLoss",
              "few_shot_samples_per_label": 10,
              "few_shot_epochs": 10,
              "few_shot_learning_rate": 0.001,
              "few_shot_batch_size": 16,
              "n_neighbors": 10,
              "image_retrieval_N_rank": 20,
              "include_baseline": False,
              }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loading model -----------------------------------------------------------------------------------------------------
    clip_model, preprocess = clip.load("ViT-B/32", download_root=os.path.join(os.path.dirname(__file__), "saved_models"))
    # Restoring checkpoints
    if np.any(params["model_checkpoints"]):
        clip_model.load_state_dict(torch.load(os.path.join(params["model_checkpoints"]), map_location=device))
    clip_model.to(device).eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    collate_fn = torch.utils.data.default_collate
    # -------------------------------------------------------------------------------------------------------------------
    
    # Datasets
    print("Loading datasets: Started...", end="\r")
    datasets = {"zero_shot":{"train":[], "val":[], "test":[], "tot":[]},
                "few_shot":{"train":[], "val":[], "test":[], "tot":[]},
                "image_retrieval":{"train":[], "val":[], "test":[], "tot":[]}}
    for split in ["train","val","test","tot"]:
        for dataset_name in params["datasets"]["zero_shot"]:
            datasets["zero_shot"][split].append(globals()[dataset_name](preprocess, split, crop=True, text_template=params["text_template"], label_type="label"))
        for dataset_name in params["datasets"]["zero_shot"]:
            if split=="train":
                datasets["few_shot"][split].append(globals()[dataset_name](preprocess, split, crop=True, label_type="label", encoder_type="one_hot_encoder", samples_per_label=params["few_shot_samples_per_label"]))
            else:
                datasets["few_shot"][split].append(globals()[dataset_name](preprocess, split, crop=True, label_type="label", encoder_type="one_hot_encoder"))
        for dataset_name in params["datasets"]["image_retrieval"]:
            datasets["image_retrieval"][split].append(globals()[dataset_name](preprocess, split, crop=True, label_type="sentence"))
    print("Loading datasets: Done      ")
    
    # Dataloaders
    print("Loading dataloaders: Started...", end="\r")
    dataloaders = {"zero_shot":{"train":[], "val":[], "test":[], "tot":[]},
                    "few_shot":{"train":[], "val":[], "test":[], "tot":[]},
                    "image_retrieval":{"train":[], "val":[], "test":[], "tot":[]}}
    for split in ["train","val","test","tot"]:
        for dataset in datasets["zero_shot"][split]:
            dataloaders["zero_shot"][split].append(DataLoader(dataset=dataset, batch_size=dataset.batch_size["zero_shot"], shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.Pool()._processes-4))
        for dataset in datasets["few_shot"][split]:
            if split=="train":
                dataloaders["few_shot"][split].append(DataLoader(dataset=dataset, batch_size=params["few_shot_batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.Pool()._processes-4))
            else:
                dataloaders["few_shot"][split].append(DataLoader(dataset=dataset, batch_size=dataset.batch_size["zero_shot"], shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.Pool()._processes-4))
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
    
    # Few-shot classification
    print("Few-shot classification: Started...")
    criterion = globals()[params["few_shot_loss"]]()
    # Iterating each datasets
    for i in range(len(datasets["few_shot"]["train"])):
        
        # Train
        dataset = datasets["few_shot"]["train"][i]
        dataloader = dataloaders["few_shot"]["train"][i]
        model = globals()["CLIP_Projection"](len(dataset.unique_labels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["few_shot_learning_rate"])
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        samples = 0
        cumulative_corrects = 0
        for epoch in range(1, params["few_shot_epochs"]+1):
            with tqdm(dataloader, unit="batch") as tepoch:
                for inputs, Y in tepoch:
                    X = inputs["image"]
                    tepoch.set_description(dataset.__class__.__name__.ljust(12)+" - "+"Train - Epoch "+str(epoch)+"/"+str(params["few_shot_epochs"]))
                    
                    # Extracting clip features
                    images = X.to(device)
                    image_features = clip_model.encode_image(images)
                    
                    # Forward pass
                    with torch.autocast("cuda"):
                        pred = model(image_features)
                    classes = Y.squeeze().type(torch.HalfTensor).to(device)
                    loss = criterion(pred, classes)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    _, pred_label_encoded = pred.max(dim=1)
                    pred_label_encoded = pred_label_encoded.to(device)
                    _, truth_label_encoded = classes.max(dim=1)
                    truth_label_encoded = truth_label_encoded.to(device)
                    
                    samples += pred.shape[0]
                    cumulative_corrects += (pred_label_encoded == truth_label_encoded).sum().item()
                    cumulative_accuracy = cumulative_corrects / samples
                    tepoch.set_postfix(loss=loss.item(), score=cumulative_accuracy)
        
        # Test
        dataset = datasets["few_shot"]["test"][i]
        dataloader = dataloaders["few_shot"]["test"][i]
        model.eval()
        samples = 0
        cumulative_corrects = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12)+" - "+"Test ")
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images)
                
                # Forward pass
                with torch.autocast("cuda"):
                    pred = model(image_features)
                classes = Y.squeeze().type(torch.HalfTensor).to(device)
                
                _, pred_label_encoded = pred.max(dim=1)
                pred_label_encoded = pred_label_encoded.to(device)
                _, truth_label_encoded = classes.max(dim=1)
                truth_label_encoded = truth_label_encoded.to(device)
                
                samples += pred.shape[0]
                cumulative_corrects += (pred_label_encoded == truth_label_encoded).sum().item()
                cumulative_accuracy = cumulative_corrects / samples
                tepoch.set_postfix(score=cumulative_accuracy)
        report_log = pd.concat([report_log, pd.DataFrame([{"dataset_name":dataset.__class__.__name__, "dataset_type":"few_shot", "accuracy":cumulative_accuracy}])]).reset_index(drop=True)
    print("Few-shot classification: Done      ")
    
    # KNN classification
    print("KNN classification: Started...")
    criterion = globals()[params["few_shot_loss"]]()
    # Iterating each datasets
    for i in range(len(datasets["few_shot"]["train"])):
        
        # Train
        dataset = datasets["few_shot"]["train"][i]
        dataloader = dataloaders["few_shot"]["train"][i]
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
        dataset = datasets["few_shot"]["test"][i]
        dataloader = dataloaders["few_shot"]["test"][i]
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
    
    # Image retrieval
    print("Image retrieval: Started...")
    for dataset, dataloader in zip(datasets["image_retrieval"]["tot"], dataloaders["image_retrieval"]["tot"]):
        samples = 0
        cumulative_corrects = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, Y in tepoch:
                X = inputs["image"]
                tepoch.set_description(dataset.__class__.__name__.ljust(12))
                
                # Extracting clip features
                images = X.to(device)
                image_features = clip_model.encode_image(images.to(device))
                text_tokens = clip.tokenize(dataset.data["sentence"].values).to(device)
                text_features = clip_model.encode_text(text_tokens)
                
                text_features = torch.div(text_features, text_features.norm(dim=1, keepdim=True))
                image_features = torch.div(image_features, image_features.norm(dim=1, keepdim=True))
                similarity_matrix = torch.matmul(text_features, image_features.t())
                
                for i in range(images.shape[0]):
                    scores_df = pd.DataFrame(similarity_matrix.cpu()[:,i])
                    scores_df["sentence"] = dataset.data["sentence"]
                    scores_df = scores_df.sort_values(by=[0], ascending=False).reset_index(drop=True)
                    if Y[i] in scores_df.iloc[:params["image_retrieval_N_rank"]]["sentence"].values:
                        cumulative_corrects += 1
                
                samples += len(X)
                cumulative_accuracy = cumulative_corrects / samples
                tepoch.set_postfix(score=cumulative_accuracy)
        report_log = pd.concat([report_log, pd.DataFrame([{"dataset_name":dataset.__class__.__name__, "dataset_type":"image_retrieval", "accuracy":cumulative_accuracy}])]).reset_index(drop=True)
    print("Image retrieval: Done      ")
    
    # Save report log
    if args.log=="True":
        if os.path.exists(os.path.join(os.path.dirname(__file__),"reports",initial_datetime)):
            shutil.rmtree(os.path.join(os.path.dirname(__file__),"reports",initial_datetime))
        os.makedirs(os.path.join(os.path.dirname(__file__),"reports",initial_datetime))
        report_log.reset_index(drop=True, inplace=True)
        report_log.to_csv(os.path.join(os.path.dirname(__file__),"reports",initial_datetime,"report_"+initial_datetime+".csv"), index=True, lineterminator='\r\n')
        json.dump(params, open(os.path.join(os.path.dirname(__file__),"reports",initial_datetime,"report_"+initial_datetime+".json"), "w"))
        build_report(params, report_log, initial_datetime)
    print("=========================================")
    print(report_log)
    print("=========================================")
    
    execution_time = int(time.time() - start_time)
    print("Execution time: ",time_convert(execution_time))