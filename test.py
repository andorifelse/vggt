import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
# 现将参数加载到内存中，在转到显存中，防止爆内存
state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)

print("模型参数加载完毕")


# Load and preprocess example images (replace with your own image paths)
image_names = ["examples/kitchen/images/00.png","examples/kitchen/images/01.png","examples/kitchen/images/02.png"]

images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

print(predictions)

