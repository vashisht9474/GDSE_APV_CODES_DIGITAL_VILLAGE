import os
import copy
import torch
import gradio
import gradio as gr
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

os.system("wget https://www.dropbox.com/s/grcragozd4x79zc/model_best.pth?dl=0")

model = torch.load("./model_best.pth?dl=0", map_location=device)

# img  = Image.open(path).convert('RGB')
from torchvision import transforms

transforms2 =  transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# img = transforms(img)
# img = img.unsqueeze(0)
model.eval()

labels = ['Tomato_Late_blight', 'Tomato_healthy', 'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Soybeanhealthy', 'Squash_Powdery_mildew', 'Potato_healthy', 'Corn(maize)Northern_Leaf_Blight', 'Tomato_Early_blight', 'Tomato_Septoria_leaf_spot', 'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry_Leaf_scorch', 'Peach_healthy', 'Apple_Apple_scab', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Bacterial_spot', 'Apple_Black_rot', 'Blueberry_healthy', 'Cherry(including_sour)Powdery_mildew', 'Peach_Bacterial_spot', 'Apple_Cedar_apple_rust', 'Tomato_Target_Spot', 'Pepper,_bell_healthy', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'PotatoLate_blight', 'Tomato_Tomato_mosaic_virus', 'Strawberry_healthy', 'Apple_healthy', 'Grape_Black_rot', 'Potato_Early_blight', 'Cherry(including_sour)healthy', 'Corn(maize)Common_rust', 'GrapeEsca(Black_Measles)', 'Raspberryhealthy', 'Tomato_Leaf_Mold', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Pepper,_bell_Bacterial_spot', 'Corn(maize)__healthy']

# with torch.no_grad():
#   # preds = 
#   preds = model(img)
#   score, indices = torch.max(preds, 1)

def recognize_digit(image):
    image = transforms2(image)
    image = image.unsqueeze(0)
    # image = image.unsqueeze(0)
    # image = image.reshape(1, -1)
    # with torch.no_grad():
    # preds = 
    # img = image.reshape((-1, 3, 256, 256))
    preds = model(image).flatten()
      # prediction = model.predict(image).tolist()[0]
    # score, indices = torch.max(preds, 1)
    # return {str(indices.item())}
    return {labels[i]: float(preds[i]) for i in range(38)}


im = gradio.inputs.Image(
    shape=(256, 256), image_mode="RGB", type="pil")

iface = gr.Interface(
    recognize_digit,
    im,
    gradio.outputs.Label(num_top_classes=3),
    live=True,
    interpretation="default",
    examples=[["images/cheetah1.jpg"], ["images/lion.jpg"]],
    capture_session=True,
)

iface.test_launch()
iface.launch()
