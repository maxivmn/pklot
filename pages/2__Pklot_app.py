## To-DO:
# - introduce error handling e.g. check all inputs for correctness. No . in id-List. Only ,

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
from modeling.detection import * 
from modeling.classification import *


st.set_page_config(page_title='ParkAI', layout='wide',
                #    initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
)
st.image("./images/ParkAI_9_transp.png",use_column_width=True)
st.header('Try it out yourself', divider='rainbow')

# To-Do: Being able to upload multiple images and then to choose multiple of them.
st.subheader('Do you want to upload a new parkling lot or use an already saved one?')


# Declaration of session state
if 'detect' not in st.session_state:
    st.session_state.detect = False
if 'correct' not in st.session_state:
    st.session_state.correct = False
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
if 'xml_string_new' not in st.session_state:
    st.session_state.xml_string_new = None

# Deviding the page in 2 columns
col1, col2 = st.columns(2)
with col1:
    file_placeholder_new = st.empty()
    uploaded_file_new = file_placeholder_new.file_uploader("Upload new, full parking lot")
    image_placeholder= st.empty()
    text_placeholder = st.empty()
with col2:
    file_placeholder_old = st.empty()
    uploaded_file_old = file_placeholder_old.file_uploader("Upload picture of existing parking lot")
    image_placeholder_old= st.empty()
    xml_placeholder = st.empty()

# Resetting the session state if a new file is uploaded.    
if (uploaded_file_new != st.session_state.file_new) | (uploaded_file_old != st.session_state.file_old):
    #Resetting the session state
    st.session_state.detect = False
    st.session_state.correct = False
    st.session_state.xml_string_new = None
    st.session_state.img_w_removed_boxes = []
    
    
    image_placeholder_old.empty()
    xml_placeholder.empty()
    image_placeholder.empty()

     
    # If new file uploaded reset history states.   
    if (uploaded_file_new != st.session_state.file_new):
        image_placeholder_old.empty()
        st.session_state.file_new = uploaded_file_new
        st.session_state.new_trigger = True
        st.session_state.old_trigger = False
    elif (uploaded_file_old != st.session_state.file_old):
        image_placeholder.empty()
        st.session_state.file_old = uploaded_file_old
        st.session_state.new_trigger = False
        st.session_state.old_trigger = True
 
 
   

#######################################
#######    If new parking lot   #######
#######################################
if st.session_state.new_trigger is True:
    
     # Reset output from old parking lot
    file_placeholder_new = st.empty()
    file_bytes = uploaded_file_new.read()
    
    # Convert the bytes to a numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Read the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption = 'Uploaded image', use_column_width=True)
    
    # 1.Detection
    button_placeholder_detect = st.empty()
    detect = button_placeholder_detect.button("Detect")
    # If detect button clicked ones, save the information.
    if detect:
        st.session_state.detect = True
    # Check if the detect button is clicked
    if st.session_state.detect:
        image_with_removed_boxes = [] 
        remove = False
        box_field = False
           
        with col1:
            button_placeholder_detect.empty()
            
            # Detect
            # Only execute when detect was pressed the first time.
            if detect: 
                prediction = detect_boxes(image)
                xml_string_det = export_to_xml(prediction)
                st.session_state.xml_string_new = xml_string_det 
                
            image_with_boxes = show_images_with_boxes(image, st.session_state.xml_string_new )
            image_placeholder.empty()
            image_placeholder.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption = 'Detected boxes',use_column_width=True)
            
            
            question_placeholder = st.empty()
            question_placeholder.write('Are the detected parking places ok or do some boxes need to be removed?')
            
            # Remove or accept buttons for detected boxes.
            # Create two buttons.
            col3, col4 = st.columns(2)
            with col3:
                button_placeholder_ok = st.empty()
                correct = button_placeholder_ok.button("Ok")
                if correct:
                    st.session_state.correct = True

            with col4:
                
                # HTML template with JavaScript to detect Enter key press
                enter_press_script = """
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    var textInput = document.querySelector('input[data-baseweb="input"]');
                    textInput.addEventListener('keypress', function(event) {
                        if (event.key === 'Enter') {
                            textInput.value = '';
                        }
                    });
                });
                </script>
                """
                # Inject the JavaScript to detect Enter key press
                text_placeholder.markdown(enter_press_script, unsafe_allow_html=True)
                button_placeholder_remove = st.empty()
                remove = button_placeholder_remove.button("Remove")
                text_placeholder = st.empty()
                box_field = text_placeholder.text_input("Box numbers to be removed", placeholder="12, 45")       
          
        # 3. Classification
        if st.session_state.correct:
            image_placeholder.empty()
            button_placeholder_detect.empty()
            button_placeholder_remove.empty()
            button_placeholder_ok.empty()
            question_placeholder.empty()
            text_placeholder.empty()
            remove = False
            box_field = False
            
            #Classification
            # Usage of images with boxes as input in order to have the ids plotted. 
            if st.session_state.img_w_removed_boxes != []:
                input = st.session_state.img_w_removed_boxes
            else:
                input = image
            
            image_out, result = run_prediction_classi(input, st.session_state.xml_string_new)
            image_placeholder.image(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB), caption = 'Classified boxes', use_column_width=True)
            st.write(f'Number of parking spaces: {result[0]}')
            st.write(f'Number of empty spaces: {result[1]}')
            st.write(f'Number of occupied spaces: {result[2]}')
            
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            clicked = st.download_button(label='Download BoxMap for Parking Lot', data= st.session_state.xml_string_new, file_name=f'BoxMap_{current_datetime}.xml')
            
            # TO-DO: Save the map in a specified path.
              
        # 2. Correction
        # Accept input if enter was pressed or if remove button as pressed and the input is not empty.
        if box_field or (remove and box_field):
            st.session_state.correct = False
            
            image_placeholder.empty()
            box_nums = [num.strip() for num in box_field.split(',')]
            box_nums_int = [int(num) for num in box_nums]


            st.session_state.xml_string_new= remove_boxes_from_xml(st.session_state.xml_string_new, box_nums_int)
            # Store the updated XML string in the session state for use in future updates
            image_with_removed_boxes = show_images_with_boxes(image, st.session_state.xml_string_new)
            st.session_state.img_w_removed_boxes = image_with_removed_boxes
            
            image_placeholder.image(cv2.cvtColor(image_with_removed_boxes, cv2.COLOR_BGR2RGB), caption = 'with deleted boxes', use_column_width=True)
        
                

            

        
       
#######################################
#######    If old parking lot   #######
#######################################
if st.session_state.old_trigger is True:
    # Reset output from new parking lot
    image_placeholder.empty()
    
    # Convert the bytes to a numpy array
    file_bytes = uploaded_file_old.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Read the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_placeholder_old.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption = 'Uploaded image', use_column_width=True)
    
    with col2:
        xml_placeholder = st.empty()
        uploaded_xml = xml_placeholder.file_uploader("Upload BoxMap of existing parking lot")
    
        if uploaded_xml is not None:   
            xml_placeholder.empty()
            xml_string_old = create_xml_string(uploaded_xml)
            image_out, result = run_prediction_classi(image, xml_string_old)
            image_placeholder_old.image(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB), caption = 'Classified boxes', use_column_width=True)
            st.write(f'Number of parking spaces: {result[0]}')
            st.write(f'Number of empty spaces: {result[1]}')
            st.write(f'Number of occupied spaces: {result[2]}')
    

    