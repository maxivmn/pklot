import sys
import os
import io
from fastapi import FastAPI
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error
import warnings
import mlflow
from mlflow.sklearn import load_model
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from feature_eng_pklot import *

app = FastAPI()

def ensemble_models(images, model_path):
    pass
    
    
# Helper functions
def load_models(path):
    pass

def visualize(image, xml_path, model_path):
    pass


#------------------------------------------------------------------
def yolo_predict(yolo_run, img_path):
    ''' 
    Give the prediction from a given yolo model for a given image
    INPUT:
        # yolo_run : path to the trained yolo model -> e.g.:  '../runs/detect/train70/
        # img_path : full path to the image want to predict

    OUTPUT
        # pred : output yolo prediction object 
    '''
    yolo_path = os.path.join(yolo_run, 'weights/best.pt')
    yolo_model = YOLO(yolo_path)
    pred = yolo_model(
        source=img_path, 
        save= False, 
        conf= 0.05,  # object confidence threshold for detection
        iou= 0.3,    # intersection over union (IoU) threshold for NMS
    )
    return pred

#------------------------------------------------------------------
def show_predict(yolo_pred_obj, image_width, image_height):
    ''' 
    Show the output image of the yolo prediction with the bounding boxes
    INPUT :
        # yolo_pred_obj : The output from yolo_predict() object

    OUTPUT :
        # Just visualize the predicted image
    '''
    num_images = len(yolo_pred_obj)  # Number of images
    canvas_width = image_width
    canvas_height = image_height * num_images

    # Create a blank canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    # Show the results for one image
    # Loop through your predictions and draw each image on the canvas
    y_position = 0
    for r in yolo_pred_obj:
        im_array = r.plot(line_width=1)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        # Paste the current image onto the canvas
        canvas.paste(im, (0, y_position))
        y_position += image_height  # Move the position for the next image
    
    # Save the final canvas to a bytes buffer
    img_byte_arr = io.BytesIO()
    canvas.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return img_byte_arr



#------------------------------------------------------------------
def create_xml_yolo(image_path, pred_result, output_path):
    ''' 
    Creates an XML file with recognized boxes.
    ARGUMENTS
    # image_path : consists of the full path whereimage is taken from -> e.g.: '../data/PKLot/Sunny_most_empty/test/images/2013-03-02_08_45_03.jpg'
    # pred_result : the prediction results comes out of yolo_predict()
    # output_path : complete ouput path for the .xml file which should be saved  e.g. : output_xml_folder = '../data/PKLot/prediction_xml_folder'

    '''
    if len(pred_result) == 0:
        print(f"No bounding boxes found in {image_path}. Skipping XML creation.")
        return
    
    image_name = image_path.split('/')[-1]  # Extracting the image name from the path
    annotation = ET.Element("annotation")
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name
    
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(pred_result.orig_shape[1])  # Assuming 'orig_shape' attribute is available in result
    height = ET.SubElement(size, "height")
    height.text = str(pred_result.orig_shape[0])
    
    for idx, box in enumerate(pred_result.boxes.xyxy):  # Access 'xyxy' attribute of the 'boxes' object
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = f"object_{idx}"  # You can customize the object names
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(box[0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(box[1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(box[2]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(box[3]))
    
    xml_str = ET.tostring(annotation, encoding="unicode")
    xml_str_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")

    with open(output_path, "w") as xml_file:
        xml_file.write(xml_str_pretty)
    print(f"XML file saved at {output_path}")
