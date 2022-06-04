import keras
import numpy as np
import requests
from PIL import Image
from io import BytesIO

model = keras.models.load_model('./model')
class_names = ['Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy', 'Potato_Early_blight', 'Potato_healthy', 'Potato_Late_blight', 'Tomato_Target_Spot', 'Tomato_Tomato_mosaic_virus', 'Tomato_Tomato_YellowLeaf_Curl_Virus', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite']

def predict(url):
    img = load_img(url)
    return predict_img(img)

def load_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img;

def predict_img(img):
    output = model.predict(img)
    ind = np.argsort(output[0])
    ind = ind[-5 :]
    results = []
    for index, val in enumerate(ind):
        obj = {}
        obj['disease'] = class_names[val]
        obj['propability'] = float(output[0][val])
        results.append(obj)
    return results