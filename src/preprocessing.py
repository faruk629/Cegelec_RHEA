import os
from sklearn.utils import Bunch
import numpy as np 
import matplotlib.pyplot as plt 
import polars as pl 
from concurrent.futures import ThreadPoolExecutor, as_completed
from bispectrum2D import Bispectrum2D
import seaborn as sns

MAX_PROFIL_LENGTH = 437


def process_profile_1D(profile):
    profile = profile.sort("x").unique(subset=['x'])
    profile = profile.filter( (pl.col("x") > -470) & (pl.col("x") < 470 ))
    profile =  profile.with_columns(((pl.col('y')+1200)/10).cast(pl.Float64))
    profile = pl.Series(profile.select('y'))
    if(len(profile) < MAX_PROFIL_LENGTH) :
        profile = np.pad(profile, (0, MAX_PROFIL_LENGTH - len(profile)), 'symmetric')
    return profile


# # def pad_to_same_size(data):
# #     max_length = max(len(d) for d in data)  # Find the maximum length
# #     return [pl.Series(tf.pad(d, paddings=[[0, max_length - len(d)]], mode='SYMMETRIC')) if len(d) < max_length else d for d in data]

def process_file(file_path):
    profil = pl.read_csv(file_path, has_header=False, new_columns=['x','y'])
    processed_profil = process_profile_1D(profil)
    return processed_profil

def sleepers_dataset_from_directory(src, dimension=1):
    """
    Converts sleepers dataset from a directory to images.

    Args:
        src (str): Path to the directory containing sleepers data.
        dimension (int, optional): Dimension of the dataset (default is 1).

    Returns:
        None: The function performs the necessary actions based on the parameters.
    """

    classes_id = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    data = []
    target = []

    # ThreadPoolExecutor allows to create a pool of threads to execute tasks in parallel.
    with ThreadPoolExecutor() as executor:
        for classe_id in classes_id:
            dataset_path = os.path.join(src, classe_id)
            files_path = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.csv')]
            futures = [executor.submit(process_file, file_path) for file_path in files_path]
            for future in as_completed(futures):
                processed_profil = future.result()
                data.append(processed_profil)
                target.append(classe_id)

    if dimension == 1:
        data = np.array(data)
    return Bunch(data=data, target=np.array(target))


def profil_to_image(profil,src,resolution):
    plt.figure(figsize=(resolution/plt.rcParams['figure.dpi'], resolution/plt.rcParams['figure.dpi']))
    sns.heatmap(np.array(profil).reshape(-1, 1), cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
    # Show the plot
    # bs2D = Bispectrum2D(signal=np.array(profil), freqsample=16_000, window_name='hanning')
    base, _ = os.path.splitext(src)
    # Change the extension to ".aln"
    new_file_path = base + ".png"
    # bs2D.plot_bispec_magnitude(new_file_path)
    plt.axis('off')  # Turn off axis
    plt.savefig(new_file_path, bbox_inches='tight', pad_inches=0) 
    plt.close()