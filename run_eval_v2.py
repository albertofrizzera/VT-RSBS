'''
This module hopefully will be a all in one evaluation. It will be used to evaluate the model on several tasks (zero-shot classification - retrieval - etc.) on different datasets.
'''
import torch
import clip
from dotenv import load_dotenv
from dataset_v2 import *
from tqdm import tqdm
from utils import recall_at_k
from sklearn.linear_model import LogisticRegression
from utils import load_remoteCLIP

remoteCLIP_models = ["RN50","ViT-B-32","ViT-L-14"]

# ALL THE PARAMETERS
load_function = load_remoteCLIP
BASE_MODEL = "remoteCLIP_RN50"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
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
    #model, preprocess = clip.load(name=BASE_MODEL, device=DEVICE)
    model, preprocess, tokenizer = load_function(BASE_MODEL, device=DEVICE)
    # Load the checkpoint 
    # checkpoint = torch.load(f"{CHECKPOINT_PATH}")
    # model.load_state_dict(checkpoint)
    model.eval()
    file = open(SAVE_REPORT_PATH, "w")
    with torch.no_grad():
        if len(ZERO_SHOT)>0:
            print("Testing zero-shot classification")
            print("#################################")
        for dataset_name in ZERO_SHOT:
            print("Testing dataset: ", dataset_name)
            dataset = globals()[dataset_name](split="test", label_type="label", preprocess=preprocess)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            # Create text templates 
            unique_labels = dataset._get_unique_labels()
            text_templates = []
            for label in unique_labels:
                text_templates.extend([t.format(label) for t in TEXT_TEMPLATES])
            
            templates = tokenizer(text_templates).to(DEVICE)
            text_features = model.encode_text(templates)
            text_features /= text_features.norm(dim=1, keepdim=True)
            
            cumulative_corrects = 0
            # Iterate over the samples
            for _, batch in enumerate(tqdm(dataloader)):
                images, labels = batch
                images = images.to(DEVICE)
                # Get the features
                image_features = model.encode_image(images)
                # Normalize them
                image_features /= image_features.norm(dim=1, keepdim=True)
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
            
            zero_shot_accuracy = cumulative_corrects / len(dataset) * 100.
            file.write(f"Zero-shot accuracy on {dataset_name} is: {round(zero_shot_accuracy,2)}\n")
        
        file.write("\n")
        
        # LINEAR PROBING
        if len(ZERO_SHOT)>0:
            print("Testing linear probing")
            print("#################################")
        for dataset_name in ZERO_SHOT:
            print("Testing dataset: ", dataset_name)
            trainset = globals()[dataset_name](split="train", label_type="label", preprocess=preprocess)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            unique_labels = trainset._get_unique_labels()
            
            train_features = []
            train_labels = []
            for batch in tqdm(trainloader):
                images, labels = batch
                images = images.to(DEVICE)
                
                image_features = model.encode_image(images)
                train_features.append(image_features.cpu())
                train_labels.append(torch.tensor([unique_labels.index(l) for l in labels], dtype=torch.long).cpu())

            train_features, train_labels = torch.cat(train_features).numpy(), torch.cat(train_labels).numpy()
            
            testset = globals()[dataset_name](split="test", label_type="label", preprocess=preprocess)
            testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            
            test_features = []
            test_labels = []
            for batch in tqdm(testloader):
                images, labels = batch
                images = images.to(DEVICE)
                
                image_features = model.encode_image(images) 
                test_features.append(image_features.cpu()) 
                test_labels.append(torch.tensor([unique_labels.index(l) for l in labels], dtype=torch.long).cpu()) # Displace on the CPU to avoid memory issues
            
            test_features, test_labels = torch.cat(test_features).numpy(), torch.cat(test_labels).numpy()
            
            # Perform logistic regression
            classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=False, n_jobs=8)
            classifier.fit(train_features, train_labels)

            # Evaluate using the logistic regression classifier
            predictions = classifier.predict(test_features)
            accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
            file.write(f"Linear-probe accuracy on {dataset_name} is: {round(accuracy,2)}\n")

        file.write("\n")
    
        # TEXT TO IMAGE RETRIEVAL
        print("Testing retrieval")
        print("#################################")
        for dataset_name in RETRIEVAL:
            print("Testing dataset: ", dataset_name)
            file.write("Testing dataset: "+ dataset_name+"\n")
            dataset = globals()[dataset_name](split="test", label_type="sentence", preprocess=preprocess)
            image, sentences = dataset[0]
            t2i, i2t = recall_at_k(model, dataset, k_vals=retrieval_k_vals, batch_size=BATCH_SIZE, device=DEVICE)

            file.write("Text-to-image Recall@K\n")
            for k, x in zip(retrieval_k_vals, t2i):
                file.write(f"   R@{k}: {100*x:.2f}%\n")

            file.write("Image-to-text Recall@K\n")
            for k, x in zip(retrieval_k_vals, i2t):
                file.write(f"   R@{k}: {100*x:.2f}%\n")
            file.write("\n")
    
    file.close()