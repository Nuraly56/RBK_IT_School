import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# === 1. Загружаем модель MobileNetV2 ===
model = models.mobilenet_v2(pretrained=True)
model.eval()  # переводим модель в режим предсказания

# === 2. Загружаем список классов ImageNet (1000 штук) ===
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = requests.get(url).text.strip().split("\n")

# === 3. Функция для загрузки и предсказания картинки ===
def predict_image(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probs, 1)

    label = classes[top_idx[0]]
    confidence = round(top_prob[0].item() * 100, 2)
    return label, confidence

images = {
    "Кот": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    "Самолёт": "https://www.aeroflap.com.br/wp-content/uploads/2020/04/69720_britisha747400_862060.jpg",
    "Телефон": "https://object.pscloud.io/cms/cms/Photo/img_0_77_2993_0_1.jpg"
}


# === 5. Предсказание для каждой картинки ===
for name, url in images.items():
    label, confidence = predict_image(url)
    print(f"{name}: {label} ({confidence}%)")

