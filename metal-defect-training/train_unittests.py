import os
import sys
from tensorflow.keras.models import Model, load_model

import numpy as np
import unittest
import yaml
from train import (
    NoDatasetError,
    DatasetPathError,
    IncorrectFormatError,
    EpochSizeError,
    SaveModelError, validate_arguments
)



class TestMetalDefect(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create artifacts for testing"""
        # Read config file for unittesting
        with open("./test_config.yaml", "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        # Define paths from config file for validate_arguments function
    
        self.train_dir = config["train_dir"]
        self.test_dir = config["test_dir"]
        self.epoch = config["epoch"]
        self.model = load_model('DefectModel')





class TestValidateArguments(TestMetalDefect):
    def test_none_dataset_error(self):
        """Checks for NoDatasetError if all directories are None"""
        with self.assertRaises(NoDatasetError):
            validate_arguments("None", "None", "30")

    def test_dataset_path_error(self):
        """Checks for DatasetPathError if one of train and test image directory is None"""
        with self.assertRaises(DatasetPathError):
            validate_arguments("None", self.test_dir, "15")
        with self.assertRaises(DatasetPathError):
            validate_arguments(self.train_dir, "None", "1")

    def epoch_size_error(self):
        """Checks for EpochSizeError if number of epochs is not between 1 and 50"""
        with self.assertRaises(DatasetPathError):
            validate_arguments(self.train_dir, self.test_dir, "55")
        with self.assertRaises(DatasetPathError):
            validate_arguments(self.train_dir, self.test_dir, "0")


class TestModelError(TestMetalDefect):
    def test_return_type(self):
        """Checks if the function returns a model h5 file"""
        self.assertIsInstance(self.model, h5)

    def __str__(self):
        return "TestModelError: Model h5 file not saved, check model output"


