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
    validate_arguments
)


class TestMetalDefect(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create artifacts for testing"""
        # Read config file for unittesting
        with open("./test_config.yaml", "r") as file:
            config1 = yaml.load(file, Loader=yaml.FullLoader)

        with open("./config_params.yaml", "r") as file:
            config2 = yaml.load(file, Loader=yaml.FullLoader)



        # Define paths from config file for validate_arguments function

        self.train_dir = config1["train_dir"]
        self.test_dir = config1["test_dir"]
        self.test_dir_txt = config1["test_dir_txt"]
        self.epoch = config1["epoch"]
        self.model = load_model("DefectModel.h5")
        self.test_data = config1["test_data"]
        self.test_labels = config1["test_labels"]
        self.train_acc = config1["train_acc"]
        self.train_loss = config1["train_loss"]
        self.test_acc = config1["test_acc"]
        self.test_loss = config1["test_loss"]


        self.train_acc_low = config2["train_al"]
        self.train_acc_high = config2["train_ah"]
        self.train_loss_low = config2["train_ll"]
        self.train_loss_high = config2["train_lh"]

        self.test_acc_low = config2["test_al"]
        self.test_acc_high = config2["test_ah"]
        self.test_loss_low = config2["test_ll"]
        self.test_loss_high = config2["test_lh"]



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


class TrainParamsError(TestMetalDefect):

    def train_params(self):
        """Checks if the model has train parameters within bounds"""
        self.assertTrue(self.train_acc_low <= self.train_acc <= self.train_acc_high & self.train_loss_low <= self.train_loss <= self.train_loss_high)

    def __str__(self):
        return "TrainParametersError: Model train accuracy/loss is not within acceptable parameters"





class TestParamsError(TestMetalDefect):

    def test_params(self):
        """Checks if the model has test parameters within bounds"""
        self.assertTrue(self.test_acc_low <= self.test_acc <= self.test_acc_high & self.test_loss_low <= self.test_loss <= self.test_loss_high)

    def __str__(self):
        return "TestParametersError: Model test accuracy/loss is not within acceptable parameters"


    

class SaveModelError(TestMetalDefect):
    def test_return_type(self):
        """Checks if the function returns a model h5 file"""
        self.assertIsInstance(self.model, h5)

    def __str__(self):
        return "SaveModelError: Model h5 file not saved, check model output"
