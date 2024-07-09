'''
This module hopefully will be a all in one evaluation. It will be used to evaluate the model on several tasks (zero-shot classification - retrieval - etc.) on different datasets.
'''
import torch
from dotenv import load_dotenv
from datasets import *
from tqdm import tqdm
from utils import recall_at_k
from sklearn.linear_model import LogisticRegression
from utils import load_CLIP, load_clipRSICDv2, load_geoRSCLIP, load_remoteCLIP
from utils import encode_text_CLIP, encode_image_CLIP, encode_text_CLIPrsicdv2, encode_image_CLIPrsicdv2, encode_text_geoRSCLIP, encode_image_geoRSCLIP, encode_text_remoteCLIP, encode_image_remoteCLIP

original_CLIP_models = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"]
remoteCLIP_models = ["RN50", "ViT-B-32", "ViT-L-14"]
geoRSCLIP_models = ["ViT-B-32", "ViT-L-14", "ViT-L-14-336", "ViT-H-14"]
clip_rsicdv2_models = ["flax-community/clip-rsicd-v2"]

# DEFINE YOUR CUSTOM FUNCTIONS TO LOAD THE MODEL AND GET THE EMBEDDINGS OUT OF IT
load_function = load_CLIP
encode_text_fn = encode_text_CLIP
encode_image_fn = encode_image_CLIP

BASE_MODEL = "CLIPrsicdv2_"+original_CLIP_models[0]
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
SAVE_REPORT_PATH = "reports/report_"+BASE_MODEL.replace("/","")+".txt"
# "UCM","WHU_RS19","RSSCN7","SIRI_WHU","RESISC45","RSI_CB128","RSI_CB256","EuroSAT","PatternNet","OPTIMAL_31","MLRSNet","RSICD","RSITMD"
ZERO_SHOT = ["UCM","WHU_RS19","RSSCN7","SIRI_WHU","RESISC45","RSI_CB128","RSI_CB256","EuroSAT","PatternNet","OPTIMAL_31","MLRSNet","RSICD","RSITMD"]
# "RSICD","RSITMD","UCM","SIDNEY"
RETRIEVAL = ["RSICD","RSITMD","UCM","SIDNEY"]
retrieval_k_vals = [1, 5, 10, 50]
#TEXT_TEMPLATES = ['a centered satellite photo of {}', 'a centered satellite photo of a {}','a centered satellite photo of the {}']
TEXT_TEMPLATES = ["a remote sensing image of a {}"]
NUM_WORKERS = 8
IMAGE_SIZE = 224

if __name__ == '__main__':
    load_dotenv()
    # Load the model
    model, textprocessor, imageprocessor = load_function(BASE_MODEL, device=DEVICE)
    model.eval()
    
    file = open(SAVE_REPORT_PATH, "w")
    with torch.no_grad():
        if len(ZERO_SHOT)>0:
            print("Testing zero-shot classification")
            print("#################################")
        for dataset_name in ZERO_SHOT:    
            print("Testing dataset: ", dataset_name)
            testset = globals()[dataset_name](split="test", label_type="label")
            testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)
            
            # Create text templates 
            unique_labels = testset._get_unique_labels()
            text_templates = []
            for label in unique_labels:
                text_templates.extend([t.format(label) for t in TEXT_TEMPLATES])
            
            # Get the text features
            text_features = encode_text_fn(model=model, textprocessor=textprocessor, texts=text_templates, device=DEVICE)
            # Normalize them
            text_features /= text_features.norm(dim=1, keepdim=True)
            
            cumulative_corrects = 0
            # Save test image features to save time in linear probing
            test_features = []
            test_labels = []
            
            for _, batch in enumerate(tqdm(testloader)):
                images, labels = batch
                # Get the image features
                image_features = encode_image_fn(model=model, imageprocessor=imageprocessor, images=images, device=DEVICE)
                
                # Append the features and labels to the lists
                test_features.append(image_features.cpu())
                test_labels.append(torch.tensor([unique_labels.index(l) for l in labels], dtype=torch.long).cpu())
                
                # Normalize them
                image_features /= image_features.norm(dim=-1, keepdim=True)
    
                # Calculate similarity
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                
                if len(TEXT_TEMPLATES)>1:
                    similarity = torch.cumsum(similarity, dim=-1)
                    index_to_select = [((i+1)*len(TEXT_TEMPLATES))-1 for i in range(len(unique_labels))]
                    
                    assert len(index_to_select) == len(unique_labels)
                    
                    similarity = similarity[:, index_to_select]
                    similarity_shifted = torch.roll(similarity, 1, dims=1)
                    similarity_shifted[:,0] = 0 
                    
                    similarity-=similarity_shifted
                
                argmax_similarity = similarity.argmax(dim=-1)
                predicted_labels = [unique_labels[i] for i in argmax_similarity]
                
                cumulative_corrects += sum([1 for i in range(len(predicted_labels)) if predicted_labels[i] == labels[i]])
            
            zero_shot_accuracy = cumulative_corrects / len(testset) * 100.
            file.write(f"Zero-shot accuracy on {dataset_name} is: {round(zero_shot_accuracy,2)}\n")
            
            # Convert the lists to numpy arrays
            test_features, test_labels = torch.cat(test_features).numpy(), torch.cat(test_labels).numpy()
        
            # LINEAR PROBING
            print("Testing linear probing")
            print("#################################")
            
            trainset = globals()[dataset_name](split="train", label_type="label")
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)

            train_features = []
            train_labels = []
            for batch in tqdm(trainloader):
                images, labels = batch
                
                image_features = encode_image_fn(model=model, imageprocessor=imageprocessor, images=images, device=DEVICE)
                
                train_features.append(image_features.cpu())
                train_labels.append(torch.tensor([unique_labels.index(l) for l in labels], dtype=torch.long).cpu())

            train_features, train_labels = torch.cat(train_features).numpy(), torch.cat(train_labels).numpy()
            
            # Perform logistic regression
            classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=False, n_jobs=8)
            classifier.fit(train_features, train_labels)

            # Evaluate using the logistic regression classifier
            predictions = classifier.predict(test_features)
            accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
            file.write(f"Linear-probe accuracy on {dataset_name} is: {round(accuracy,2)}\n")

            file.write("\n")
        
        file.write("\n")
    
        # TEXT TO IMAGE RETRIEVAL
        print("Testing retrieval")
        print("#################################")
        for dataset_name in RETRIEVAL:
            print("Testing dataset: ", dataset_name)
            file.write("Testing dataset: "+ dataset_name+"\n")
            dataset = globals()[dataset_name](split="test", label_type="sentence")
            image, sentences = dataset[0]
            
            t2i, i2t = recall_at_k(model=model, 
                                   dataset=dataset,
                                   encode_img_fn=encode_image_fn,
                                   imageprocessor=imageprocessor,
                                   encode_text_fn=encode_text_fn,
                                   textprocessor=textprocessor,
                                   k_vals=retrieval_k_vals, 
                                   batch_size=BATCH_SIZE, 
                                   device=DEVICE)

            file.write("Text-to-image Recall@K\n")
            for k, x in zip(retrieval_k_vals, t2i):
                file.write(f"   R@{k}: {100*x:.2f}%\n")

            file.write("Image-to-text Recall@K\n")
            for k, x in zip(retrieval_k_vals, i2t):
                file.write(f"   R@{k}: {100*x:.2f}%\n")
            file.write("\n")
    
    file.close()