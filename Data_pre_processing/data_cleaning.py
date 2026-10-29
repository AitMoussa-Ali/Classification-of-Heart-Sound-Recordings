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
    data.columns = ["file", "class", "sure", "folder name"]
    return counts, data

def devide_x_y(data : pd.DataFrame) : 
    Y = data.iloc[:,1].values
    X = data.iloc[:, [0,2,3]].values

    return X,Y

def split_train_test(data : pd.DataFrame):
    data_training = data[data["folder name"] != "training-e"].reset_index(drop=True)
    data_testing = data[data["folder name"] == "training-e"].reset_index(drop=True)
    number_class_normal = (data_training["class"] == -1).sum()
    number_class_abnormal = (data_training["class"] == 1).sum()

    print("Normal:", number_class_normal)
    print("Abnormal:", number_class_abnormal)

    return data_training, data_testing