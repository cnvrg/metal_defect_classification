import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input
import os
import argparse
import matplotlib.pyplot as plt

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


# train_inclusion_dir = os.path.join('../NEU-DET/train/images/inclusion')
# train_inclusion_names = os.listdir(train_inclusion_dir)

class NoDatasetError(Exception):
    """Raise if train and validation images paths are None"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "NoDatasetError: No dataset provided. Train and validation images path cannot be None"


class DatasetPathError(Exception):
    """Raise if either the train images directory or test images directory is None"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return (
            "DatasetPathError: The tain images directory or test images directory cannot be None"
        )


    
class EpochSizeError(Exception):
    """Raise if epochs size is not between 0 and 50"""

    def __init__(self, epoch):
        super().__init__(epoch)
        self.epoch = epoch

    def __str__(self):
        return f"EpochSizeError: {self.epoch} is an invalid size. Number of epochs needs to be a value between 1 and 50"


class IncorrectFormatError(Exception):
    """Raise if images are not in .jpg or .png"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "IncorrectFormatError: Train and test images should be in .jpg or .png format"


    
def validate_arguments(
    train_directory, test_directory, epoch
):
    """Validates input arguments
    Checks if the input arguments provided by the user are appropriate for running this library.
    
    Args:
        train_directory: The directory containing training images
        test_directory: The directory containing test images
        epoch: Number of epochs to run VGG19 model
    Raises:
        NoDatasetError: If train_directory and test_directory are both None
        DatasetPathError: If either train images directory or test images directory is None
        IncorrectFormatError: If the dataset files are not in jpg or png format
        EpochSizeError: If epoch size is not between 1 and 50
             

    """
    if (
        train_directory.lower() == "none"
        and test_directory.lower() == "none"
    ):
        raise NoDatasetError



    if train_directory.lower() == "none":
        if (test_directory.lower() != "none" 
        ):
            raise DatasetPathError



    if not any(ext in train_directory and test_directory for ext in [".png", ".jpg"]):
        raise IncorrectFormatError


    if not (epoch > 0 and epoch <= 50):
        raise EpochSizeError(epoch)



def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""Metal Defect Training""")
    parser.add_argument(
        "--train_path",
        action="store",
        dest="train_path",
        required=False,
        default="../NEU-DET/train/images/",
        help="""--- Path to the original training dataset ---""",
    )
    parser.add_argument(
        "--test_path",
        action="store",
        dest="test_path",
        required=False,
        default="../NEU-DET/validation/images/",
        help="""--- Path to the original test dataset ---""",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        required=False,
        default=cnvrg_workdir,
        help="""--- The path to save model file to ---""",
    )
    parser.add_argument(
        "--epoch",
        action="store",
        dest="test_path",
        required=False,
        default=10,
        help="""--- Number of epochs to train VGG19 model ---""",
    )

    return parser.parse_args()


def define_model():
    # load model
    model = VGG19(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
    output = Dense(6, activation="sigmoid")(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def prepare_dataset(train_path, test_path):

    # rescale images 
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # create training images 
    train_generator = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",
    )

    # create validation images
    val_generator = train_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",
    )

    return train_generator, val_generator


def train_vgg19(model, train_generator, val_generator, epoch):
    
    history = model.fit(
        train_generator,
        steps_per_epoch=16,
        epochs=epoch,
        verbose=1,
        validation_data=val_generator,
        shuffle=True,
    )

    # Plot accuracy
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    # Plot loss
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    return model


def metal_defect_train_main():
    args = parse_parameters()

    model = define_model()
    model.summary()

    train_generator, val_generator = prepare_dataset(args.train_path, args.test_path)

    model_new = train_vgg19(model, train_generator, val_generator, args.epoch)

    model_new.save(args.output_dir + "DefectModel.h5")


if __name__ == "__main__":
        metal_defect_train_main()
