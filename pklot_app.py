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
from datetime import datetime

import cv2
from PIL import Image
from modeling.predict_pklot import * 
from modeling.detect_pklot import *


st.set_page_config(page_title='ParkAI', layout='wide',
                #    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)
st.image("./images/ParkAI_9_transp.png",use_column_width=True)


# Declaration of session state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
if 'correct' not in st.session_state:
    st.session_state.correct = False
if 'remove' not in st.session_state:
    st.session_state.remove = False
if 'file_new' not in st.session_state:
    st.session_state.file_new = None
if 'file_old' not in st.session_state:
    st.session_state.file_old = None
if 'img_w_removed_boxes' not in st.session_state:
    st.session_state.img_w_removed_boxes = []
if 'new_trigger' not in st.session_state:
    st.session_state.new_trigger = False
if 'old_trigger' not in st.session_state:
    st.session_state.old_trigger = False


st.title('Try it out yourself')

# To-Do: Being able to upload multiple images and then to choose multiple of them.
st.write('Do you want to upload a new parkling lot or use an already saved one?')


col1, col2 = st.columns(2)
with col1:
    file_placeholder_new = st.empty()
    uploaded_file_new = file_placeholder_new.file_uploader("Upload new, full parking lot")
    image_placeholder= st.empty()
with col2:
    file_placeholder_old = st.empty()
    uploaded_file_old = file_placeholder_old.file_uploader("Upload picture of existing parking lot")
    image_placeholder_old= st.empty()
    xml_placeholder = st.empty()

button_placeholder_d = st.empty()
image_placeholder_c = st.empty()
# question_placeholder = st.empty()
# col1, col2 = st.columns(2)
# # Add a button to the first column
# with col1:
#     button_placeholder_p = st.empty()

# Resetting the session state if a new file is uploaded.    
if (uploaded_file_new != st.session_state.file_new) | (uploaded_file_old != st.session_state.file_old):
    
    #Resetting the session state
    st.session_state.button_clicked = False
    st.session_state.correct = False
    st.session_state.remove = False
    st.session_state.img_w_removed_boxes = []
    
    image_placeholder_old.empty()
    xml_placeholder.empty()
    button_placeholder_d.empty()
    image_placeholder.empty()
    #image_placeholder_old.empty()
        
    if (uploaded_file_new != st.session_state.file_new):
        #file_placeholder_o.empty()
        image_placeholder_old= st.empty()
        st.session_state.file_new = uploaded_file_new
        st.session_state.new_trigger = True
        st.session_state.old_trigger = False
        #st.session_state.file_new = uploaded_file_new
    elif (uploaded_file_old != st.session_state.file_old):
        #file_placeholder_n.empty()
        image_placeholder= st.empty()
        st.session_state.file_old = uploaded_file_old
        st.session_state.new_trigger = False
        st.session_state.old_trigger = True
    


###############################
###    If new parking lot   ###
###############################  
if st.session_state.new_trigger is True:
    
    # New path
    file_placeholder_new = st.empty()
    file_bytes = uploaded_file_new.read()
    # Convert the bytes to a numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Read the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img = Image.open(uploaded_file)
    
    image_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption = 'Uploaded image', width=500)
    
    button_clicked = button_placeholder_d.button("Detect")
    if button_clicked:
        st.session_state.button_clicked = True
    # Check if the button is clicked
    if st.session_state.button_clicked:
        
        #button_placeholder.empty()
        #st.write("Button clicked!")
        #image_path ='./data/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_06_05_16.jpg'
        #image = cv2.imread(image_path)
        
        
        # 1. Detection

        prediction = detect_boxes(image)
        xml_string_det = export_to_xml(prediction)
        if 'xml_string_det' not in st.session_state:
            st.session_state.xml_string_det = xml_string_det   
        
        image_with_boxes = show_images_with_boxes(image, xml_string_det)
        image_with_removed_boxes = []
        image_placeholder_c = st.empty()
        image_placeholder_c.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption = 'Detected boxes')
        
        question_placeholder = st.empty()
        question_placeholder.write('Are the detected parking places ok or do some boxes need to be removed?')
        
        # Create two buttons.
        col1, col2 = st.columns(2)
        # Add a button to the first column
        with col1:
            button_placeholder_p = st.empty()
            correct = button_placeholder_p.button("Ok")
            if correct:
                st.session_state.correct = True
                st.session_state.remove = False

        # Add a button to the second column
        with col2:
            button_placeholder_c = st.empty()
            remove = button_placeholder_c.button("Remove")
            text_placeholder = st.empty()
            box_field = text_placeholder.text_input("Box numbers to be removed", placeholder="12, 45")
            if remove:
                st.session_state.remove = True
            
        # 2. Correction
        if remove:
            #image_placeholder_c.empty()
            box_nums = [num.strip() for num in box_field.split(',')]
            box_nums_int = [int(num) for num in box_nums]

            # If it's the first time removing boxes, use the original XML string.
            # Otherwise, use the XML string from the last update.
            if 'updated_xml_string' not in st.session_state:
                st.session_state.updated_xml_string = xml_string_det 

            box_removed_xml_string= remove_boxes_from_xml(st.session_state.updated_xml_string, box_nums_int)
            # Store the updated XML string in the session state for use in future updates
            st.session_state.updated_xml_string = box_removed_xml_string

            image_with_removed_boxes = show_images_with_boxes(image, box_removed_xml_string)
            st.session_state.img_w_removed_boxes = image_with_removed_boxes
            #image_placeholder_c = st.empty()
            image_placeholder_c.image(cv2.cvtColor(image_with_removed_boxes, cv2.COLOR_BGR2RGB), caption = 'with deleted boxes')
        
                
        # 3. Classification
        if st.session_state.correct:
            st.session_state.remove = False
            button_placeholder_d.empty()
            button_placeholder_c.empty()
            button_placeholder_p.empty()
            question_placeholder.empty()
            #image_placeholder_c.empty()
            text_placeholder.empty()
            
            #Classification
            #st.write('hello')
            # Useage of images with boxes as input in order to have the ids plotted. 
            if st.session_state.img_w_removed_boxes != []:
                input = st.session_state.img_w_removed_boxes
                xml_string = st.session_state.updated_xml_string
            else:
                input = image
                xml_string = st.session_state.xml_string_det 
            
            image_out, result = run_prediction_classi(input, xml_string)
            image_placeholder_c.image(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB), caption = 'Classified boxes')
            st.write(f'Number of parking spaces: {result[0]}')
            st.write(f'Number of empty spaces: {result[1]}')
            st.write(f'Number of occupied spaces: {result[2]}')
            
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            clicked = st.download_button(label='Download Boxmap for Parking Lot', data= xml_string, file_name=f'BoxMap_{current_datetime}.xml')
            
            # TO-DO: Save the map in a specified path.
            
            

        
       
###############################
###    If new parking lot   ###
###############################
if st.session_state.old_trigger is True:
    # Reset all history states connected to new path
    image_placeholder.empty()
    button_placeholder_d.empty()
    
    file_bytes = uploaded_file_old.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Read the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img = Image.open(uploaded_file)
    
    image_placeholder_old.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption = 'Uploaded image', width=500)
    
    with col2:
        xml_placeholder = st.empty()
        uploaded_xml = xml_placeholder.file_uploader("Upload BoxMap of existing parking lot")
    
    if uploaded_xml is not None:   
        xml_string_old = create_xml_string(uploaded_xml)
        image_out, result = run_prediction_classi(image, xml_string_old)
        image_placeholder_c.image(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB), caption = 'Classified boxes')
        st.write(f'Number of parking spaces: {result[0]}')
        st.write(f'Number of empty spaces: {result[1]}')
        st.write(f'Number of occupied spaces: {result[2]}')
    

    