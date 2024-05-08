from huggingface_hub import hf_hub_download

for model_name in ['ViT-H-14', 'ViT-B-32', 'ViT-L-14', 'ViT-L-14-336']:
    checkpoint_path = hf_hub_download("Zilun/GeoRSCLIP", f"ckpt/RS5M_{model_name}.pt", cache_dir='geoRSCLIP')
    print(f'{model_name} is downloaded to {checkpoint_path}.')