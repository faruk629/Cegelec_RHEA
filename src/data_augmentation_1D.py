import numpy as np
import polars as pl 

# Rotate the signal 
def rotate(signal, degree, random=False):
    """
    Rotate the signal by the specified degree.

    Parameters:
        signal (DataFrame): The signal data to be rotated.
        degree (float): The degree by which to rotate the signal.
        random (bool, optional): If True, a random degree between -degree and degree will be generated (default is False).

    Returns:
        Series: The rotated signal.
    """

    signal = pl.DataFrame(signal).clone()

    if random and degree < 0:
        raise ValueError("Degree must be positive when random is True.")
    

    # Add index column
    signal = signal.with_columns(pl.lit(np.arange(0,len(signal),1)).alias('index'))
    # Translate the signal's index to the center
    center = signal.select('index').mean()
    translated_signal = signal.with_columns((pl.col('index') - center).cast(pl.Float64))

    if (random):
        # Generate a random degree between -degree and degree
        degree = np.random.uniform(-degree, degree)

    # # Perform rotation around the center
    theta = degree * np.pi / 180
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))


    index = pl.Series(translated_signal.select('index'))
    column = pl.Series(translated_signal.select('column_0'))
    # # Rotate each point individually
    rotated_values = np.dot(R, np.vstack((index, column)))

    return rotated_values[1]
 



def shift_data_y(signal): 
    return signal.copy() + np.random.uniform(-5, 5)

def shift_data_x(signal):
    shift_amount = np.random.randint(1, 30)  # Random shift amount between 1 and 20
    shift_direction = np.random.choice([1, -1])  # Randomly select shift direction (1 for right, -1 for left)
    
    if shift_direction == 1:  # Shift to the right
       shifted_data = pl.DataFrame(signal).shift(shift_amount,fill_value=pl.DataFrame(signal)[1]).select('column_0')
    else:  # Shift to the left
       shifted_data = pl.DataFrame(signal).shift(-shift_amount,fill_value=pl.DataFrame(signal)[-1]).select('column_0')
    
    return shifted_data.to_numpy().flatten()

def add_noise(signal):
    # Add random noise to the signal
    noise_level =0.1 # Adjust noise level as needed
    noise = np.random.normal(scale=noise_level, size=signal.shape)
    noisy_signal = signal.copy() + noise
    return noisy_signal


def random_augmentation(signal):

    # Generate random True or False values for each augmentation technique
    random_rotate = np.random.choice([True, False])
    random_shift_x = np.random.choice([True, False])
    random_shift_y = np.random.choice([True, False])
    random_noise = np.random.choice([True, False])

    augmented_x = signal.copy()

    # Ensure at least one augmentation technique is selected
    if not (random_rotate or random_shift_x or random_noise or random_shift_y):
        random_noise = True

    # Apply each augmentation technique based on the generated random values
    if random_rotate:
        augmented_x = rotate(augmented_x, 0.2,True)
    if random_shift_x:
        augmented_x = shift_data_x(augmented_x)
    if random_shift_y:
        augmented_x = shift_data_y(augmented_x)
    if random_noise:
        augmented_x = add_noise(augmented_x)

    return augmented_x

def augment(data, target):
    augmented_x = data.copy()
    augmented_y = target.copy()

    augmentations = [
        ('vertical flip', lambda x: np.flip(x).flatten()), # 4400
        ('horizontal flip', lambda x: rotate(x, 180)), # 8800
        ('rotation', lambda x: rotate(x, 0.9, random=True)), # 17600
        ('shift_y', lambda x: shift_data_y(x)), # 35520
        ('shift_x', lambda x: shift_data_x(x)), # 71040
        ('noise', lambda x: add_noise(x)), # 142080
        ('random', lambda x: random_augmentation(x)) # 284160
    ]

    for name, augmentation in augmentations:
        augmented_data = [augmentation(x_).reshape(-1,1) for x_ in augmented_x]
        augmented_x = np.concatenate((augmented_x, augmented_data), axis=0)
        augmented_y = np.vstack([augmented_y, augmented_y])
    return augmented_x, augmented_y
