import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import cv2
from keras import layers, models
from keras import applications
from keras import callbacks
from pycocotools import mask as maskUtils

class ImageSegmentationModel:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        base_model = applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        base_model.trainable = True

        for layer in base_model.layers[:20]:
            layer.trainable = False

        conv1 = base_model.get_layer('block2a_expand_activation').output
        conv2 = base_model.get_layer('block3a_expand_activation').output
        conv3 = base_model.get_layer('block4a_expand_activation').output
        conv4 = base_model.get_layer('block6a_expand_activation').output
        conv5 = base_model.get_layer('top_activation').output

        up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
        up6 = layers.concatenate([up6, conv4], axis=3)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
        up7 = layers.concatenate([up7, conv3], axis=3)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
        up8 = layers.concatenate([up8, conv2], axis=3)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
        up9 = layers.concatenate([up9, conv1], axis=3)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

        up10 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv9)
        conv10 = layers.Conv2D(32, 3, activation='relu', padding='same')(up10)
        conv10 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv10)

        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv10)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        return model
    
    def preprocess_data(self, annotation):
        size = annotation['size']
        mask_encoded = annotation['counts']
        mask_encoded = np.array2string(annotation['counts'].numpy())
        mask_encoded = mask_encoded[3:-2]
        rle = {'size': size, 'counts': mask_encoded.replace('\\\\','\\')}
        mask = maskUtils.decode(rle)
        return mask
    
    def load_dataset(self, dataset_name='test_coco_dataset', split='train'):
        return tfds.load(dataset_name, split=split)
    
    def prepare_data(self, dataset):
        images = []
        image_id_to_index = {}
        image_id_to_mask = {}
        masks = []

        for idx, example in enumerate(dataset):
            img_id = example['image_id'].numpy()
            image = example['image'].numpy()
            image = cv2.resize(image, (256, 256))
            images.append(image)
            image_id_to_index[img_id] = idx

        images = np.array(images)
        masks = np.zeros((len(images), 256, 256), dtype=np.uint8)

        for example in dataset:
            img_id = example['image_id'].numpy()
            segmentation = example['segmentation']
            mask = self.preprocess_data(segmentation)
            mask = cv2.resize(mask, (256, 256))

            if img_id in image_id_to_mask:
                existing_mask = image_id_to_mask[img_id]
                combined_mask = np.maximum(existing_mask, mask)
                image_id_to_mask[img_id] = combined_mask
            else:
                image_id_to_mask[img_id] = mask

        masks = [image_id_to_mask[example['image_id'].numpy()] for example in dataset]

        masks = np.array(masks).astype(np.float32)
        images = images / 255.0
        
        return images, masks
    
    @staticmethod
    def dice_loss(y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        return 1. - score
    
    def data_generator(self, images, masks, batch_size):
        while True:
            for start in range(0, len(images), batch_size):
                end = min(start + batch_size, len(images))
                yield images[start:end], masks[start:end]

    def train(self, train_images, train_masks, val_images, val_masks, epochs=3, batch_size=8):
        model = self.model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=self.dice_loss, metrics=['accuracy'])
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model_checkpoint = callbacks.ModelCheckpoint('unet_model2.keras', save_best_only=True, monitor='val_loss')

        train_gen = self.data_generator(train_images, train_masks, batch_size)
        val_gen = self.data_generator(val_images, val_masks, batch_size)
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=len(train_images) // batch_size,
            validation_steps=len(val_images) // batch_size,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        return history
    
    def plot_training(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    
    def display_prediction(self, image, mask, prediction, threshold=0.5):
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


# Использование класса
segmentation_model = ImageSegmentationModel()
dataset = segmentation_model.load_dataset()

train_images, train_masks = segmentation_model.prepare_data(dataset)
train_images, train_masks = train_images[:600], train_masks[:600]

val_images, val_masks = segmentation_model.prepare_data(dataset)
val_images, val_masks = val_images[600:], val_masks[600:]

history = segmentation_model.train(train_images, train_masks, val_images, val_masks)
segmentation_model.plot_training(history)
val_predictions = segmentation_model.model.predict(val_images)

for i in range(5):
    segmentation_model.display_prediction(val_images[i], val_masks[i].squeeze(), val_predictions[i].squeeze())