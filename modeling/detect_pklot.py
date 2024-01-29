from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

def detect_boxes(image):
    # do the prediction with confidence level of maximum 0.2; iou allows for intersection of predicted boxes (lower values are stricter)
    model = YOLO('./models/best.pt')
    # force model to use cpu even if trained on gpu
    results = model(image, conf=0.05, iou=0.25, device='cpu', classes=1) # 1=cars only for transfer learning model 
    return results[0]

def export_to_xml(prediction, display_output=False, testing=False, write_xml=True):
    '''
    Creates an XML-structure with coordinates of the boxes.
    '''
    
    # Create the XML structure
    root = ET.Element("parking", id='/Users/margarita.samuseva/neuefische/pklot/data')
    contours = []
    labels = []
    id_counter = 1
    for box in prediction.boxes:
        # Create the <space> which equals to one parking spot
        label = int(box.cls.item())
        space = ET.SubElement(root, "space", id=str(id_counter), occupied=str(label), confidence=str(round(box.conf[0].item(), 4)))
        # Get the cords in xywh format
        cords = box.xywh[0].tolist()
        x, y, w, h = [round(x) for x in cords]
        # Get the cords in xyxy format
        cords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = [round(x) for x in cords]
        # Rotated rectangle, that is not rotated - just keeping this for naming convention
        rotated_rect = ET.SubElement(space, "rotatedRect")
        ET.SubElement(rotated_rect, "center", x=str(x), y=str(y))
        ET.SubElement(rotated_rect, "size", w=str(w), h=str(h))
        # Contour
        contour = ET.SubElement(space, "contour")
        ET.SubElement(contour, "point", x=str(x1), y=str(y1))
        ET.SubElement(contour, "point", x=str(x2), y=str(y1))
        ET.SubElement(contour, "point", x=str(x2), y=str(y2))
        ET.SubElement(contour, "point", x=str(x1), y=str(y2))
        contours.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        labels.append(label)
        id_counter += 1
    
    # format XML and make it more readable    
    prettify_xml(root)
    
    # Convert the XML structure to a string
    xml_string = ET.tostring(root, encoding='unicode')
    if display_output:
        print("XML content generated successfully.")
        return xml_string
    if testing:
        return contours, labels, xml_string
    else:
        return xml_string

## Helper functions.

# make the xml code more readable
def prettify_xml(element, indent='  '):
    queue = [(0, element)]  # (level, element)
    while queue:
        level, element = queue.pop(0)
        children = list(element)
        if children:
            element.text = '\n' + indent * (level+1)  # for child open
        if queue:
            element.tail = '\n' + indent * queue[0][0]  # for sibling open
        else:
            element.tail = '\n' + indent * (level-1)  # for parent close
        queue[0:0] = [(level + 1, child) for child in children]

# Helper functions
def remove_boxes_from_xml(xml_string, ids_to_remove):
    ''' 
    Removes boxes from XML file.
    Creates an XML file with reduced numer of recognized boxes.
    '''
    # Parse the XML string
    tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()

    # Iterate over 'space' elements and remove those with matching IDs
    for space in root.findall('.//space'):
        if int(space.get('id')) in ids_to_remove:
            root.remove(space)

    # Return the modified XML as a string
    return ET.tostring(root, encoding='unicode')



def run_detection(image):
        
    prediction = detect_boxes(image)
    xml_str = export_to_xml(prediction)
    return prediction, xml_str

if __name__ == "__main__":
    image_path ='/Users/margarita.samuseva/neuefische/pklot/data/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_06_05_16.jpg'
    image = cv2.imread(image_path)
    image_new, result = run_detection(image)

    