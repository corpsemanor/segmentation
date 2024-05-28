"""test_coco_dataset dataset."""

import shutil
import tensorflow_datasets as tfds
from test_coco_dataset_dataset_builder import Builder

class TestCocoDatasetTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for test_coco_dataset dataset."""
    DATASET_CLASS = Builder
    SPLITS = {
        'train': 3,  
        'test': 1,   
    }

    def setUp(self):
        super().setUp()
        # Clear the directory used for testing
        test_dir = tfds.testing.test_utils.test_src_dir(self.__class__)
        shutil.rmtree(test_dir, ignore_errors=True)

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests need to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}

if __name__ == '__main__':
    tfds.testing.test_main()
