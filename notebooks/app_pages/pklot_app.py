## To-DO:
# read data from a link
# predict. Write function apply model.
# plot an answer.

import sys
import os

# Add PKLOT directory to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


import torch
import streamlit as st
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
from modeling.predict import *


st.set_page_config(page_title='ParkAI', layout='wide',
                #    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)
#st.image("./images/ParkAI_2.png",use_column_width=True)
#st.snow()
st.title('Try it out yourself')

#st.write('hello world')

col1, col2 = st.columns(2)

with col1:
    # To-Do: Being able to upload multiple images and then to choose multiple of them.
    uploaded_file = st.file_uploader("Choose an image")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, width=500)


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
            st.write("Button clicked!")
            # Perform actions or computations here  
    else:
        st.text('Upload image to analyze')
