## To-DO:
# read data from a link
# predict. Write function apply model.
# plot an answer.

import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder #responsible for loading images from train and test folders
from torch.utils.data import DataLoader, random_split, ConcatDataset # Definition of iterable
from torchvision import transforms, models # Provides image transformations
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score

import shutil
import seaborn as sns
import cv2
from PIL import Image
from modeling.predict_pklot import * 
from modeling.detect_pklot import *


st.set_page_config(page_title='ParkAI', layout='wide',
                #    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)
st.image("./images/ParkAI_9_transp.png",use_column_width=True)
#st.snow()

if 'button_clicked' not in st.session_state:
    st.session_stat.button_clicked = False
if 'correct' not in st.session_state:
    st.session_state.correct = False
if 'remove' not in st.session_state:
    st.session_state.remove = False



st.title('Try it out yourself')

# To-Do: Being able to upload multiple images and then to choose multiple of them.
#file_placeholder = st.empty()
uploaded_file = st.file_uploader("Choose an image")

#st.session_state.clear()

if uploaded_file is not None:
    #file_placeholder = st.empty()
    file_bytes = uploaded_file.read()
    # Convert the bytes to a numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Read the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img = Image.open(uploaded_file)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption = 'Uploaded image', width=500)
    
    st.session_state.button_clicked = st.button("Detect")
    # Check if the button is clicked
    if st.session_state.button_clicked:
        #button_placeholder.empty()
        #st.write("Button clicked!")
        #image_path ='./data/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_06_05_16.jpg'
        #image = cv2.imread(image_path)
        
        ## If new parking lot
        # 1. Detection
        prediction = detect_boxes(image)
        xml_string = export_to_xml(prediction)
        image_with_boxes = show_images_with_boxes(image, xml_string)
        image_1= st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption = 'Detected boxes')
        
        st.write('Are the detected parking places ok or do some boxes need to be removed?')
        
        # Create two buttons.
        col1, col2 = st.columns(2)

        # Add a button to the first column
        with col1:
            st.session_state.correct = st.button("Ok")

        # Add a button to the second column
        with col2:
            st.session_state.remove = st.button("Remove")
        
        # 2. Correction
        # while remove:
        #     pass
            
            # 3. Classification
        if st.session_state.correct:
            #Classification
            #st.write('hello')
            image_out, result = run_prediction_classi(image, xml_string)
            st.image(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB), caption = 'Detected boxes')
            st.write(result[0])
        
else:
    st.text('Upload image to analyze')

#def analyze_image(model, )


