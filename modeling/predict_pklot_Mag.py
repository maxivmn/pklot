import sys
from fastapi import FastAPI
from sklearn.metrics import mean_squared_error
import warnings
import mlflow
from mlflow.sklearn import load_model
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

from detect_pklot import *
from feature_eng_pklot import *
from config import *

app = FastAPI()
     
# Helper functions    
def load_model(model_path):
    # Extrahiere den Modellnamen aus dem Dateipfad
    model_name = model_path.split('/')[-1].split('_')[0].lower()

    # Dynamisch das entsprechende Modell laden
    if 'squeezenet' in model_name:
        loaded_model = models.squeezenet1_1(pretrained=False)
        loaded_model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        loaded_model.load_state_dict(torch.load(model_path))
    elif 'resnet' in model_name:
        loaded_model = models.resnet18()
        loaded_model.fc = nn.Linear(loaded_model.fc.in_features, 2)
        loaded_model.load_state_dict(torch.load(model_path))
    elif 'mobile' in model_name:
        loaded_model = models.mobilenet_v3_small()
        loaded_model.classifier[3] = nn.Linear(loaded_model.classifier[3].in_features, 2) 
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()        
    else:
        raise ValueError(f"Unrecognized model: {model_name}")

    loaded_model.eval()

    return loaded_model

def ensemble_and_classify(cropped_images, model_paths): # TO-DO: Umbenennen. Predict in Namen reinbrignen.
    num_parking_spaces = len(cropped_images)

    # Laden Sie die drei Modelle
    loaded_models = [load_model(model_path) for model_path in model_paths]

    data_transform = transforms.Compose([
        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH), interpolation=Image.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    num_empty_spaces = 0
    num_occupied_spaces = 0
    empty_space_ids = []
    occupied_space_ids = []

    with torch.no_grad():
        for box_info in cropped_images:
            image = box_info['image']

            # Transformiere das Bild
            image_transformed = data_transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            image_tensor = torch.unsqueeze(image_transformed, 0)

            # Durchlaufe jedes Modell und speichere die Wahrscheinlichkeiten
            probabilities = []
            for loaded_model in loaded_models:
                outputs = loaded_model(image_tensor)
                probabilities.append(outputs.softmax(dim=1).numpy())

            # Berechne den Durchschnitt der Wahrscheinlichkeiten
            average_probabilities = np.mean(probabilities, axis=0)

            # Klassifiziere anhand des Durchschnitts
            prediction = np.argmax(average_probabilities)

            # Füge die Klassifizierung zum to_classify-Datensatz hinzu
            box_info['status'] = 'occupied' if prediction == 1 else 'empty'

            # Zählen Sie die Anzahl der leeren und besetzten Parkplätze im Batch
            if prediction == 0:
                num_empty_spaces += 1
                empty_space_ids.append(box_info['id'])
            else:
                num_occupied_spaces += 1
                occupied_space_ids.append(box_info['id'])

    #print(f'Number of parking spaces: {num_parking_spaces}\nNumber of empty spaces: {num_empty_spaces}\nNumber of occupied spaces: {num_occupied_spaces}')
    #print(f'Empty space IDs: {empty_space_ids}\nOccupied space IDs: {occupied_space_ids}')

    return num_parking_spaces, num_empty_spaces, num_occupied_spaces, empty_space_ids, occupied_space_ids


def visualize_result(image, cropped_images, result ): # TO-DO: Input original image without boxes.
    '''
    Visualization of the boxes inclusively the predicted class
    '''
    # Visualize results with marked IDs
    num_parking_spaces, num_empty_spaces, num_occupied_spaces, empty_space_ids, occupied_space_ids = result

    for box_info in cropped_images:
        space_id = box_info['id']
        contour_np = box_info['contour']

        if space_id in occupied_space_ids:
            cv2.polylines(image, [contour_np], isClosed=True, color=(0, 0, 255), thickness=2)
        elif space_id in empty_space_ids:
            cv2.polylines(image, [contour_np], isClosed=True, color=(0, 255, 0), thickness=2)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    return image



    
def run_prediction_classi(image, xml_string):
    model_paths = ['../models/resnet18_sunny_datasets.pth', '../models/squeezenet1_sunny_datasets.pth']
    #xml_path = '/Users/margarita.samuseva/neuefische/pklot/data/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_06_05_16.xml'
    cropped_images, image_with_boxes = crop_images(image, xml_string)
    result = ensemble_and_classify(cropped_images, model_paths)
    image_with_boxes_classified = visualize_result(image, cropped_images, result)
    return image_with_boxes_classified, result

if __name__ == "__main__":
    image_path ='/Users/thisal_weerasekara/neuefische/pklot/data/PKLot/PKLot/UFPR05/Cloudy/2013-03-15/2013-03-15_08_30_02.jpg'
    image = cv2.imread(image_path)
    prediction = detect_boxes(image)
    xml_string = export_to_xml(prediction)
    image_new, result = run_prediction_classi(image, xml_string)