import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import preprocessing
import cv2
from pycocotools import mask as maskUtils
import tensorflow as tf
import json

def preprocess_data(annotation):
    size = annotation['segmentation']['size']
    mask_encoded = annotation['segmentation']
    mask = maskUtils.decode(mask_encoded)
    return mask

def load_data(json_path, image_dir, image_size=(256, 256)):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = []
    image_id_to_index = {}
    for idx, image_info in enumerate(data['images']):
        file_name = image_info['file_name']
        file_name = f"{image_dir}/{file_name}"
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file '{file_name}' not found.")
        image = cv2.resize(image, image_size)
        images.append(image)
        image_id_to_index[image_info['id']] = idx

    images = np.array(images)
    annotations = data['annotations']
    masks = np.zeros((len(images), image_size[0], image_size[1]), dtype=np.uint8)

    for annotation in annotations:
        image_id = annotation['image_id']
        mask = preprocess_data(annotation)
        mask = cv2.resize(mask, image_size)
        idx = image_id_to_index[image_id]
        masks[idx] = np.maximum(masks[idx], mask)

    images = images / 255.0
    masks = np.expand_dims(masks, axis=-1)
    return images, masks

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1. - score

@tf.keras.utils.register_keras_serializable()
def custom_dice_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred)

def display_prediction(image, mask, prediction, threshold=0.96):
    prediction = (prediction > threshold).astype(np.uint8)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(mask, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(prediction, cmap='gray')
    plt.show()

def visualize_results(model_path, json_path, image_dir):
    images, masks = load_data(json_path, image_dir, image_size=(256, 256))
    model = models.load_model(model_path, custom_objects={'dice_loss': custom_dice_loss})
    val_predictions = model.predict(images)
    for i in range(5):
        display_prediction(images[i], masks[i].squeeze(), val_predictions[i].squeeze())

visualize_results('unet_model2.keras', 'instances_750.json', 'train')
