import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM
import os 
import json
def save_yolo_segmentation(masks, image_shape, class_id, f):
    height, width = image_shape
    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = contour.reshape(-1, 2)
            normalized_points = points.astype(float) / [width, height]
            line = f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])
            f.write(line + '\n')

# def save_json_labels(masks, image_shape, class_name, image_path, output_file):
#     height, width = image_shape
#     json_data = {
#         "version": "0.4.15",
#         "flags": {},
#         "shapes": [],
#         "imagePath": os.path.basename(image_path),
#         "imageData": None,
#         "imageHeight": height,
#         "imageWidth": width,
#         "text": ""
#     }
    
#     shape_id = 0
#     for class_masks in masks:
#         for mask in class_masks:
#             contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             for contour in contours:
#                 points = contour.reshape(-1, 2).tolist()
                
#                 # Create shape dictionary
#                 shape = {
#                     "label": class_name.lower(),
#                     "text": "",
#                     "points": points,
#                     "group_id": None,
#                     "shape_type": "polygon",
#                     "flags": {}
#                 }
                
#                 json_data["shapes"].append(shape)
#                 shape_id += 1
    
#     # Save JSON file
#     return json_data

def save_json_labels(masks, image_shape, image_path):
    height, width = image_shape
    json_data = {
        "version": "0.4.15",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "text": ""
    }
    
    shape_id = 0
    for i, masks in enumerate(all_masks):
        class_name = classes[i]
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print("objects: ",len(contours))
            for contour in contours:
                # Convert points to float and normalize them
                points = contour.reshape(-1, 2).astype(float)
                
                # Check if any points are outside the image boundaries
                points[:, 0] = np.clip(points[:, 0], 0, width)
                points[:, 1] = np.clip(points[:, 1], 0, height)
                
                # Convert points to list for JSON serialization
                points = points.tolist()
                
                # Create shape dictionary
                shape = {
                    "label": class_name.lower(),
                    "text": "",
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                
                json_data["shapes"].append(shape)
                shape_id += 1
    
    return json_data

def plot_segmentation(image, all_masks):
    #colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
    colors = np.array([
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255]     # Blue
    ], dtype=np.uint8)
    result = image.copy()
    
    # Overlay masks on the image
    for i, masks in enumerate(all_masks):
        color = colors[i].tolist()
        for mask in masks:
            result = cv2.addWeighted(result, 1, (mask[..., None] * color).astype(np.uint8), 0.5, 0)
    
    # for i, class_name in enumerate(classes):
    #     color = colors[i].tolist()
    #     cv2.putText(result, class_name, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return result

# Load the model and image
model = LangSAM()
classes = ['Person', 'Truck','Asphalt']

imagesPath = '../exp/images/'

labelPath = '../exp/labels/'
checkPath = '../exp/check/'

os.makedirs(labelPath, exist_ok=True) 
os.makedirs(checkPath, exist_ok=True) 

images = [file for file in os.listdir(imagesPath) if file.endswith('.png')]

for img in images:
    image_path = imagesPath+img
    image_pil = Image.open(image_path).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    all_masks = []
    labelName = img.split(".")[0]+'.json'
    for i, text_prompt in enumerate(classes):
        results = model.predict([image_pil], [text_prompt])
        masks = results[0]['masks']
        all_masks.append(masks)

    json_data = save_json_labels(all_masks, image_cv2.shape[:2],image_path)
    with open(labelPath+labelName, 'w') as f:
        json.dump(json_data, f, indent=2)
        #save_yolo_segmentation(masks, image_cv2.shape[:2], i, f)
    # f.close()

    result_image = plot_segmentation(image_cv2, all_masks)

    cv2.imwrite(checkPath+img, result_image)

