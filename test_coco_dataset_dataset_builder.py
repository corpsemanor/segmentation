import tensorflow_datasets as tfds
import tensorflow as tf
import os
import json
from pycocotools import mask as maskUtils

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_custom_coco_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self):
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="COCO compatible Synthetic Dataset",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3)),
                'label': tfds.features.ClassLabel(names=[
                    'drink_whippingcream_lucerne', 'lotion_essentially_nivea',
                    'craft_yarn_caron_01', 'cereal_cheerios_honeynut',
                    'candy_minipralines_lindt', 'pasta_lasagne_barilla',
                    'drink_greentea_itoen', 'snack_granolabar_naturevalley',
                    'snack_biscotti_ghiott_01', 'cleaning_snuggle_henkel'
                ]),
                'bbox': tfds.features.BBoxFeature(),
                'segmentation': tfds.features.FeaturesDict({
                    'size': tfds.features.Tensor(shape=(2,), dtype=tf.int64),
                    'counts': tfds.features.Sequence(tfds.features.Tensor(shape=(), dtype=tf.string)),
                }),
                'area': tfds.features.Tensor(shape=(), dtype=tf.int64),
                'iscrowd': tfds.features.Tensor(shape=(), dtype=tf.int64),
                'category_id': tfds.features.Tensor(shape=(), dtype=tf.int64),
                'image_id': tfds.features.Tensor(shape=(), dtype=tf.int64)
            }),
            supervised_keys=('image', 'segmentation'),
            homepage='https://dataset-homepage/',
            citation=r"""""",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = 'D:/DocuSketch/jun_dataset_coco/coco'
        instances_file = os.path.join(data_dir, 'instances_750.json')

        return {
            'train': self._generate_examples(instances_file, os.path.join(data_dir, 'train')),
            'test': self._generate_examples(instances_file, os.path.join(data_dir, 'test')),
        }

    def _generate_examples(self, instances_file, images_dir):
        """Yields examples."""
        with open(instances_file, 'r', encoding='utf-8') as f:
            instances = json.load(f)

        categories = {cat['id']: cat['name'] for cat in instances['categories']}
        images = {img['id']: img for img in instances['images']}
        annotations = instances['annotations']

        for i, annotation in enumerate(annotations):
            image_info = images[annotation['image_id']]
            image_path = os.path.join(images_dir, image_info['file_name'])
            label = categories[annotation['category_id']]

            bbox = annotation['bbox']
            bbox = tfds.features.BBox(
                ymin=bbox[1] / image_info['height'],
                xmin=bbox[0] / image_info['width'],
                ymax=(bbox[1] + bbox[3]) / image_info['height'],
                xmax=(bbox[0] + bbox[2]) / image_info['width']
            )

            segmentation = annotation['segmentation']
            rle_size = segmentation['size']
            rle_counts = segmentation['counts']

            if isinstance(rle_counts, str):
                rle_counts = [rle_counts]

            area = annotation.get('area', 0)
            iscrowd = annotation.get('iscrowd', 0)
            category_id = annotation.get('category_id')
            image_id = annotation.get('image_id')

            hashed_key = f'{image_info["file_name"]}_{i}'
            
            yield hashed_key, {
                'image': image_path,
                'label': label,
                'bbox': bbox,
                'segmentation': {
                    'size': rle_size,
                    'counts': rle_counts
                },
                'area': area,
                    'iscrowd': iscrowd,
                    'category_id': category_id,
                    'image_id': image_id
                }

if __name__ == '__main__':
    builder = Builder()
    builder.download_and_prepare()