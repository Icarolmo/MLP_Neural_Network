from PIL import Image
import numpy as np
import os
from queue import Queue

def _getImageData(imagePath):
    image = Image.open(f"train_data\X_png\{imagePath}")
    
    image_array = np.array(image)  
    
    flat_array = image_array.flatten()
    
    flat_array = flat_array.astype(int)
    
    flat_array = np.where(flat_array == 0, -1, flat_array)
    
    return flat_array

def _transferFilesNameForQueue(directoryPath):
    queue = Queue()
    
    for filename in os.listdir(directoryPath):
        queue.put(filename)

    return queue
    
def _getAllImageData(queue):
    x_train = []
     
    while not queue.empty():
        image_data = _getImageData(queue.get())
        
        x_train.append(image_data)
    
    return x_train
        
    
def get(directoryPath):
    queue = _transferFilesNameForQueue(directoryPath)
    x_train_data = _getAllImageData(queue)
    
    return x_train_data

def getXtrainData(directory):
    data_content = getAllData(directory)
    
    data_content_vectors = splitDataForVectors(data_content)

    return data_content_vectors
        
def getAllData(directory):
    with open(directory, 'r') as file:
        data_str = file.read().replace('\n', '')  # Remove todos os caracteres de quebra de linha
        filtered_data = data_str.replace(' ', '')
        data_vectors = [int(data) for data in filtered_data.split(',') if data]  # Remove strings vazias e armazena em vetor
        
    return data_vectors

def splitDataForVectors(data_content_vector, vector_size=120):
    vectors = []
    for i in range(0, len(data_content_vector), vector_size):
        vector = data_content_vector[i:i+vector_size]
        vectors.append(vector)
        
    return vectors
      
def getYTrainData(filePath):
    with open(filePath, 'r') as file:
        data_str = file.read()
        data_vectors = [str(data) for data in data_str.split('\n') if data] 
    
    return data_vectors
        