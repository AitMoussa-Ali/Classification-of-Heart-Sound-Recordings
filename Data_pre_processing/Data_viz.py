import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_path = r"C:\Users\aliai\Desktop\M2 I2A\Optimisation des modeles - Ouzar\Projet\Dataset\annotations\updated"
training_folders_names = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
file_name = "REFERENCE_withSQI.csv"

def count_number_of_classes(data : pd.DataFrame, index : int, class_ref : int) -> int :
    counts = data[data[index] == class_ref].shape[0]
    return counts

def data_cleaning(training_folders : list[str] = training_folders_names, file_name : str = file_name) : 
    counts = []
    data = np.empty((0, 4))  # 0 rows, 3 columns
    for training_folder in training_folders : 
        path = base_path + f"\\{training_folder}\\{file_name}"
        data_file = pd.read_csv(path, header=None)
        data_file["training_set"] = training_folder
        data = np.append(data,np.array(data_file),axis=0)
        data_count = {
        "number of normal" : count_number_of_classes(data_file, 1, 1) ,
        "number of abnormal" : count_number_of_classes(data_file, 1, -1) ,
        "number of unsure" : count_number_of_classes(data_file, 2, 0),
        "total sure" : count_number_of_classes(data_file, 2, 1),
        "training set" : training_folder
        }

        counts.append(data_count)
        
    data = pd.DataFrame(data)
    data = data[data[2] == 1]

    return counts, data
