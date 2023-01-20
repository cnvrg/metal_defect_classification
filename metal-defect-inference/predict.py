import os
import yaml
from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
from imageio import imread
import base64


lib_path = "/cnvrg_libraries/dev-metal-defect-inference/"
if os.path.exists("/cnvrg_libraries/metal-defect-inference"):
    lib_path = "/cnvrg_libraries/metal-defect-inference/"

# Read config file
with open(lib_path + "library.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def predict(data):

    
    #image = imread(data)
    model = load_model('DefectModel.h5')
    decoded = base64.b64decode(data["media"][0])
    nparr = np.fromstring(decoded, np.uint8)

    
    new_arr = resize(nparr,(224, 224))
    test = np.expand_dims(new_arr, axis=0)
    result = model.predict(test)
    result = np.argmax(result)

    result_dict = {0: 'Crazing', 1: 'Inclusion', 2: 'Patches', 3: 'Pitted Surface', 4: 'Rolled In Scale', 5: 'Scratches'}

    
    pred = {}    
    pred['Predicted Defect'] = result_dict[result]
    

    return pred









