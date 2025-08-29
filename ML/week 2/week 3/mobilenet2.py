import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = requests.get(url).text.strip().split("\n")

model = models.mobilenet_v2(pretrained=True)
model.eval()

def prepare_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_resized = image.resize((224, 224))  
    image_np = np.array(image_resized).astype(np.float32)
    image_np = image_np / 127.5 - 1.0
    image_np = np.transpose(image_np, (2, 0, 1))  # C H W
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
    return image_tensor, image  

image_urls = [
    "https://storage-api.petstory.ru/resize/1000x1000x80/11/bd/5c/11bd5cbe96e74c83a5054794065c21d4.jpeg",
    "https://www.aeroflap.com.br/wp-content/uploads/2020/04/69720_britisha747400_862060.jpg",
    "https://object.pscloud.io/cms/cms/Photo/img_0_77_2993_0_1.jpg"
]

for i, url in enumerate(image_urls, 1):
    input_tensor, original_image = prepare_image(url)

    with torch.no_grad():
        predictions = model(input_tensor)

    predicted_index = predictions[0].argmax().item()
    predicted_label = classes[predicted_index]
    confidence = torch.softmax(predictions[0], dim=0)[predicted_index].item() * 100

    plt.figure(figsize=(4, 4))
    plt.imshow(original_image)
    plt.title(f"{predicted_label} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
