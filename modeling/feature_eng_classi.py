import warnings
import xml.etree.ElementTree as ET
import cv2
import numpy as np

warnings.filterwarnings("ignore")

def crop_images(image, xml_string):
    '''
    Reads the attributes of detected bounding boxes from the xml.
    Cuts the bounding boxes from the image.
    OUTPUT:
    to_classify - list of cropped images.
    image_with_boxes - original image with boxes on top. 
    
    '''

    # XML-Datei analysieren
    # tree = ET.parse(xml_path)
    # root = tree.getroot()
    
    root = ET.fromstring(xml_string)

    # Leeres Array für die ausgeschnittenen Boxen
    cropped_images = []

    # Kopiere das Originalbild für die Anzeige der Boxen
    image_with_boxes = image.copy()

    # Iteriere durch jede Box im XML
    for space in root.iter('space'):
        space_id = int(space.attrib['id'])

        # Extrahiere Koordinaten und Winkel aus XML (Konturkoordinaten)
        contour_points = []
        for point in space.iter('point'):
            x = int(point.attrib['x'])
            y = int(point.attrib['y'])
            contour_points.append((x, y))

        # Konvertiere die Konturpunkte in ein NumPy-Array
        contour_np = np.array(contour_points, dtype=np.int32)
        contour_np = contour_np.reshape((-1, 1, 2))

        # Zeichne ein Rechteck um die Konturpunkte auf dem Bild mit Boxen
        cv2.polylines(image_with_boxes, [contour_np], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Extrahiere Winkel aus XML
        try:
            angle = float(space.find('./rotatedRect/angle').attrib['d'])
        except (AttributeError, ValueError):
            # Falls angle nicht extrahiert werden kann oder ein ValueError auftritt, setze angle auf 0
            angle = 0.0
        
        # Extrahiere Zentrum aus XML
        center_x = float(space.find('./rotatedRect/center').attrib['x'])
        center_y = float(space.find('./rotatedRect/center').attrib['y'])
        center = (center_x, center_y)

        # Berechne die Rotationsmatrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotiere das Bild
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Schneide die Box aus dem rotierten Bild aus
        rect = cv2.boundingRect(contour_np)
        box_image = rotated_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

        # Füge die ausgeschnittene Box zur Liste hinzu
        cropped_images.append({'image': box_image, 'id': space_id, 'contour': contour_np})
        
        # Beschrifte das Bild mit der Box-ID
        cv2.putText(image_with_boxes, str(space_id), (rect[0], rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # # Anzeigen des Bildes mit den eingezeichneten Boxen im Output
    # plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    return cropped_images, image_with_boxes

def create_xml_string(xml_path):
    
    # Analyze XML-file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Convert the root element to a string
    xml_string = ET.tostring(root, encoding='utf8', method='xml').decode()
    
    return xml_string
