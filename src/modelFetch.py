import pyrebase
from src.preProcess import preProcess
import numpy as np
import tensorflow as tf
import os

def modelImageFetch(imageName, modelName, firstNum, secondNum):
    firebaseConfig = {
        "apiKey": "AIzaSyBnR2KYud3YIFXhJ63oIkEI4KCrSO-tKu8",
        "authDomain": "parkinson-demo.firebaseapp.com",
        "projectId": "parkinson-demo",
        "storageBucket": "parkinson-demo.appspot.com",
        "messagingSenderId": "453871934145",
        "appId": "1:453871934145:web:b14de15a8e0f1e35961240",
        "databaseURL":"",
    }
    #Firebase initialization
    firebase = pyrebase.initialize_app(firebaseConfig)
    firebase = firebase.storage()

    #Model fetch
    modelPath = f"./data/models/{modelName}"
    firebase.child(f"models/{modelName}").download("",modelPath)

    #Fetching of the image to be tested
    imgPath = f"./data/testingImages/{imageName}"
    firebase.child(f"testing/{imageName}").download("",imgPath)
    processedImage = preProcess(imgPath, firstNum, secondNum)

    #Fetching the modal
    loaded_model = tf.keras.models.load_model(modelPath)
    prediction = loaded_model.predict(processedImage)
    classes_x = np.argmax(prediction, axis=1)

    #Remove the downloaded file to prevent server overload on future loads
    os.remove(modelPath)
    os.remove(imgPath)

    return classes_x