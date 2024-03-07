import os
import pandas as pd 
from sklearn.utils import Bunch
import numpy as np 
import tensorflow as tf



def process_profile_1D(profile):
    sorted_profile = profile.sort_values(by='x').drop_duplicates().copy()
    filtred_profile = sorted_profile[(sorted_profile.x >= -470) & (sorted_profile.x <= 470)].copy()
    filtred_profile.loc[:, 'y'] = (filtred_profile['y'] + 1200) / 10
    return filtred_profile.y



def pad_to_same_size(data):
    """
    Ajoute un padding aux données pour qu'elles aient toutes la même taille, en utilisant un mode de padding symétrique.
    """
    max_length = max(len(d) for d in data)  # Trouve la longueur maximale
    return [pd.Series(tf.pad(d, paddings=[[0, max_length - len(d)]], mode='SYMMETRIC')) if len(d) < max_length else d for d in data]

def sleepers_dataset_from_directory(src, dimension=1):
    """
    Charge tous les profils depuis le répertoire source, les traite et les regroupe avec leurs étiquettes correspondantes.

    Paramètres :
        src (str): Source des profils.
        dimension (int): Dimension du tableau. Par défaut, 1.

    Renvoie :
        numpy.ndarray: Profils chargés.
    """
    classes_id = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    data = []
    target = []
    for classe_id in classes_id:
        dataset_path = os.path.join(src, classe_id)
        files_path = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.csv')]
        for file_path in files_path:
            points = pd.read_csv(file_path, names=['x', 'y'])
            if dimension ==1 :
                processed_points = process_profile_1D(points)

            data.append(processed_points)
            target.append(classe_id)

    padded_data = pad_to_same_size(data)
    if(dimension ==1):
        padded_data = np.array(padded_data)
    return Bunch(data=padded_data, target=np.array(target))