import numpy as np
import pandas as pd
from Data_pre_processing.data_cleaning import data_cleaning, training_folders_names
import os
import soundfile as sf

# for index, row in data.iterrows():
base_path = r"C:\Users\aliai\Desktop\M2 I2A\Optimisation des modeles - Ouzar\Projet\Dataset"

def split_signals_and_build(data: pd.DataFrame, segment_duration : int = 5, expected_sample_rate : int = 2000):

    segment_size = segment_duration * expected_sample_rate
    
    segments_data = []

    for _, row in data.iterrows():
        path = os.path.join(
            base_path,
            row["folder name"],
            row["file"] + ".wav"
        )

        signal, sample_rate = sf.read(path)

        # Ensure the recording has the expected sampling rate
        if sample_rate != expected_sample_rate:
            raise ValueError(
                f"{path}: expected {expected_sample_rate} Hz, "
                f"got {sample_rate} Hz"
            )


        for start in range(0, len(signal), segment_size):
            segment = signal[start:start + segment_size]

            original_length = len(segment)

            if original_length < segment_size:
                segment = np.pad(
                    segment,
                    (0, segment_size - original_length),
                    mode="constant",
                    constant_values=0
                )

            # Keep only complete 5-second segments
            if len(segment) == segment_size:
                segments_data.append({
                    "file": row["file"],
                    "class": row["class"],
                    "sure": row["sure"],
                    "folder name": row["folder name"],
                    "start_sample": start,
                    "signal": segment
                })

    data_augmented = pd.DataFrame(segments_data)

    return data_augmented