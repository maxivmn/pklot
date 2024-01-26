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
from modeling.predict_pklot import * # this doesn't work. I don't know why. 


st.set_page_config(page_title='ParkAI', layout='wide',
                #    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)
st.image("./images/ParkAI_2.png",use_column_width=True)
#st.snow()
st.title('Try it out yourself')

#st.write('hello world')

col1, col2 = st.columns(2)

with col1:
    # To-Do: Being able to upload multiple images and then to choose multiple of them.
    uploaded_file = st.file_uploader("Choose an image")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        # Convert the bytes to a numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        # Read the image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img = Image.open(uploaded_file)
        st.image(image, width=500)
        


with col2:
    # Create a placeholder for the button
    button_placeholder = st.empty()
    st.markdown(
    """
    <style>
    .stButton > button {
        position: absolute;
        top: 300px;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    if uploaded_file is not None:
        button_clicked = button_placeholder.button("Analyze")
        # Check if the button is clicked
        if button_clicked:
            button_placeholder.empty()
            #st.write("Button clicked!")
            #image_path ='./data/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_06_05_16.jpg'
            #image = cv2.imread(image_path)
            image_new, result = run_prediction_classi(image)
            st.image(image_new, width=500)
            
    else:
        st.text('Upload image to analyze')

#def analyze_image(model, )