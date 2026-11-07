from Data_pre_processing.data_cleaning import data_cleaning, split_train_test
from Data_pre_processing.Signal_treatement import split_signals_and_build
import numpy as np


def upload_data(segment_duration : int):
    counts, data = data_cleaning()

    data = data.reset_index(drop=True)

    data_training, data_testing = split_train_test(data=data)

    #Applying the split to the training set
    data_training_splited = split_signals_and_build(data=data_training, segment_duration=segment_duration)

    signals_dict = data_training_splited.groupby('file')['signal'].apply(lambda x: np.array(x.tolist())).to_dict()
    dataset_training_x = (list(signals_dict.values()))
    dataset_training_y = np.int32(data_training.iloc[:,1].values)
    dataset_training_y = (dataset_training_y > 0).astype(np.int64)

    #Applying the split to the test set
    data_test_splited = split_signals_and_build(data=data_testing, segment_duration=5)
    signals_dict_test = data_test_splited.groupby('file')['signal'].apply(lambda x: np.array(x.tolist())).to_dict()
    dataset_testing_x = (list(signals_dict_test.values()))
    dataset_testing_y = np.int32(data_testing.iloc[:,1].values)
    dataset_testing_y = (dataset_testing_y > 0).astype(np.int64)

    return (dataset_training_x, dataset_training_y), (dataset_testing_x, dataset_testing_y)

