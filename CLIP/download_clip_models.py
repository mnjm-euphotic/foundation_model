import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(clip.available_models())

models_2_download = [ 'RN50', 'RN50x64', 'ViT-B/32', 'ViT-L/14@336px' ]
for model_name in models_2_download:
# for model_name in clip.available_models():
    model, preprocess = clip.load(model_name, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print('Downloaded', model_name, 'Params', total_params)
