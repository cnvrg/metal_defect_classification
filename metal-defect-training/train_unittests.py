import os
import sys
from tensorflow.keras.models import model, load_model


import numpy as np
import unittest
import yaml
from train import (
    NoDatasetError,
    DatasetPathError,
    IncorrectFormatError,
    EpochSizeError,
    validate_arguments,
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
        self.test_dir_txt = config["test_dir_txt"]
        self.epoch = config["epoch"]
        self.model = load_model("DefectModel.h5")
        self.test_data = config["test_data"]
        self.test_labels = config["test_labels"]


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
        with self.assertRaises(EpochSizeError):
            validate_arguments(self.train_dir, self.test_dir, "55")
        with self.assertRaises(EpochSizeError):
            validate_arguments(self.train_dir, self.test_dir, "0")


    def test_class_file_format_error(self):
        """Checks for IncorrectFormatError if images are not in jpg or png format"""
        with self.assertRaises(IncorrectFormatError):
            validate_arguments(self.train_dir, self.test_dir_txt, "3")


class ModelParamsError(TestMetalDefect):
    def __str__(self):
        """Checks if the model has parameters within bounds"""
        score = model.evaluate(self.test_images, self.test_labels, verbose=0)

        if not (float(score[0]) > 0.1 and float(score[0]) <= 0.3):
            return "ModelLossError: Model loss is not within acceptable parameters"
        if not (float(score[1]) > 88 and float(score[0]) <= 100):
            return (
                "ModelAccuracyError: Model accuracy is not within acceptable parameters"
            )


class SaveModelError(TestMetalDefect):
    def test_return_type(self):
        """Checks if the function returns a model h5 file"""
        self.assertIsInstance(self.model, h5)

    def __str__(self):
        return "SaveModelError: Model h5 file not saved, check model output"
