import unittest
import numpy as np
import tensorflow as tf
from keras import backend as K
from new_main import ImageSegmentationModel

class TestImageSegmentationModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = ImageSegmentationModel()
        cls.dataset = cls.model.load_dataset()
        cls.images, cls.masks = cls.model.prepare_data(cls.dataset)

    def setUp(self):
        self.model = self.__class__.model
        self.images = self.__class__.images
        self.masks = self.__class__.masks

    def test_build_model(self):
        model = self.model.model
        self.assertEqual(len(model.layers), 442)
        self.assertEqual(model.input_shape, (None, 256, 256, 3))

    def test_preprocess_data(self):
        example = next(iter(self.dataset))
        mask = self.model.preprocess_data(example['segmentation'])
        self.assertEqual(mask.shape, (example['size'][0], example['size'][1]))
        self.assertTrue(np.issubdtype(mask.dtype, np.bool_))

    def test_prepare_data(self):
        images, masks = self.model.prepare_data(self.dataset)
        self.assertEqual(images.shape[1:], (256, 256, 3))
        self.assertEqual(masks.shape[1:], (256, 256))
        self.assertTrue(np.issubdtype(images.dtype, np.float32))
        self.assertTrue(np.issubdtype(masks.dtype, np.float32))

    def test_dice_loss(self):
        y_true = np.random.randint(0, 2, (5, 256, 256, 1)).astype(np.float32)
        y_pred = np.random.random((5, 256, 256, 1)).astype(np.float32)
        y_true_tf = K.variable(y_true)
        y_pred_tf = K.variable(y_pred)
        loss = self.model.dice_loss(y_true_tf, y_pred_tf)
        self.assertTrue(np.issubdtype(loss.dtype, np.float32))

    def test_train(self):
        images, masks = self.images[:16], self.masks[:16]
        history = self.model.train(images, masks, images, masks, epochs=1, batch_size=4)
        self.assertIn('accuracy', history.history)
        self.assertIn('loss', history.history)

    def test_data_generator(self):
        images, masks = self.images, self.masks
        gen = self.model.data_generator(images, masks, batch_size=4)
        batch_images, batch_masks = next(gen)
        self.assertEqual(batch_images.shape, (4, 256, 256, 3))
        self.assertEqual(batch_masks.shape, (4, 256, 256))

    def test_predict_in_batches(self):
        images = self.images[:32]
        predictions = self.model.predict_in_batches(images, batch_size=8)
        self.assertEqual(predictions.shape[1:], (256, 256, 1))
        self.assertTrue(len(predictions) == len(images))

    def test_display_prediction(self):
        images, masks = self.images[:1], self.masks[:1]
        prediction = np.random.random((1, 256, 256, 1)).astype(np.float32)
        self.model.display_prediction(images[0], masks[0], prediction[0])

if __name__ == '__main__':
    unittest.main()
