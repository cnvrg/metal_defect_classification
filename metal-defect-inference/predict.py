import os
import yaml
from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
from imageio import imread
import base64
import pathlib
import sys
from skimage.color import gray2rgb 

#lib_path = "/cnvrg_libraries/dev-metal-defect-inference/"
#if os.path.exists("/cnvrg_libraries/metal-defect-inference"):
#    lib_path = "/cnvrg_libraries/metal-defect-inference/"
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
#current_dir = str(pathlib.Path(__file__).parent.resolve())
#os.environ['CURRENT_DIR']=current_dir

from prerun import download_model_files
download_model_files()

def predict(data):
    currpath = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists("/input/metal-defect training/DefectModel.h5"):
        #class_names = "/input/pose_classify/class_names.csv"
        model_path = "/input/metal-defect training/DefectModel.h5"
    else:
        model_path = os.path.join(currpath,"DefectModel.h5")
        #model_path = "DefectModel.h5"
    # Read config file
    #with open(lib_path + "library.yaml", "r") as file:
    #    config = yaml.load(file, Loader=yaml.FullLoader)
    #image = imread(data)
    model = load_model(model_path)
    decoded = base64.b64decode(data["img"][0])
    nparr = np.fromstring(decoded, np.uint8)

    
    new_arr = resize(nparr,(224, 224))
    
    # Convert grayscale image to RGB (assuming the model expects RGB input)
    new_arr_rgb = gray2rgb(new_arr)

    test = np.expand_dims(new_arr_rgb, axis=0)  # Add the batch dimension
    
    # test = np.expand_dims(new_arr, axis=0)
    result = model.predict(test)
    result = np.argmax(result)

    result_dict = {0: 'Crazing', 1: 'Inclusion', 2: 'Patches', 3: 'Pitted Surface', 4: 'Rolled In Scale', 5: 'Scratches'}

    
    pred = {}    
    pred['Predicted Defect'] = result_dict[result]
    

    return pred









