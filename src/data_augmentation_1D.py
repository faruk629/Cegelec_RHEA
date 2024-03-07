import numpy as np
import pandas as pd



# Rotate the signal 
def rotate(signal, degree, random=False):
    # Translate the signal's index to the center
    center = np.mean(signal.index)
    translated_signal = signal.copy()
    translated_signal.index = translated_signal.index - center

    if (random):
      # Generate a random degree between -0.2 and 0.2
      degree = np.random.uniform(-degree, degree)
    # Perform rotation around the center
    theta = degree * np.pi / 180
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Rotate each point individually
    rotated_values = np.dot(R, [translated_signal.index, translated_signal])

    # Translate the rotated signal back to its original position
    rotated_index = rotated_values[0] + center

    return pd.Series(rotated_values[1], index=rotated_index)



def shift_data_y(signal): 
    return signal.copy() + np.random.uniform(-5, 5)

def shift_data_x(signal):
    shift_amount = np.random.randint(1, 21)  # Random shift amount between 1 and 20
    shift_direction = np.random.choice([1, -1])  # Randomly select shift direction (1 for right, -1 for left)
    
    if shift_direction == 1:  # Shift to the right
        shifted_data = np.concatenate((np.zeros(shift_amount), signal.iloc[:-shift_amount]))
    else:  # Shift to the left
        shifted_data = np.concatenate((signal.iloc[shift_amount:], np.zeros(shift_amount)))
    
    return shifted_data

def add_noise(signal):
    # Add random noise to the signal
    noise_level =0.05 # Adjust noise level as needed
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
        augmented_x = rotate(augmented_x, 0.9,True)
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
        ('vertical flip', lambda x: np.flip(x)), # 4400
        ('horizontal flip', lambda x: rotate(x, 180)), # 8800
        ('rotation', lambda x: rotate(x, 0.9, random=True)), # 17600
        ('shift_y', lambda x: shift_data_y(pd.Series(x))), # 35520
        ('shift_x', lambda x: shift_data_x(pd.Series(x))), # 71040
        ('noise', lambda x: add_noise(pd.Series(x))), # 142080
        ('random', lambda x: random_augmentation(pd.Series(x))) # 284160
    ]

    for name, augmentation in augmentations:
        augmented_data = [np.array(augmentation(pd.Series(x_.flatten()))).reshape(-1, 1) for x_ in augmented_x]
        augmented_x = np.concatenate((augmented_x, augmented_data), axis=0)
        augmented_y = np.append(augmented_y, augmented_y)
        
    return augmented_x, augmented_y