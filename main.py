import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM
import os 

def save_yolo_segmentation(masks, image_shape, class_id, output_file):
    height, width = image_shape
    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = contour.reshape(-1, 2)
            normalized_points = points.astype(float) / [width, height]
            line = f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])
            f.write(line + '\n')

def plot_segmentation(image, all_masks, classes):
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

imagesPath = '../sim_data/images/'

labelPath = '../sim_data/labels/'
checkPath = '../sim_data/check/'
os.makedirs(labelPath, exist_ok=True) 
os.makedirs(checkPath, exist_ok=True) 

images = [file for file in os.listdir(imagesPath) if file.endswith('.png')]

for img in images:
    image_path = imagesPath+img
    image_pil = Image.open(image_path).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    all_masks = []
    labelName = img.split(".")[0]+'.txt'
    f = open(labelPath+labelName,'w')
    for i, text_prompt in enumerate(classes):
        results = model.predict([image_pil], [text_prompt])
        masks = results[0]['masks']
        all_masks.append(masks)
        save_yolo_segmentation(masks, image_cv2.shape[:2], i, f)
    f.close()

    result_image = plot_segmentation(image_cv2, all_masks, classes)

    cv2.imwrite(checkPath+img, result_image)

