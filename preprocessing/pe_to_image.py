import numpy as np
from PIL import Image

def binary_to_image(pe_file, width=256):
    with open(pe_file, 'rb') as f:
        data = f.read()

    length = len(data)
    height = np.ceil(length / width).astype(int)
    padded_length = width * height
    data_padded = np.pad(np.frombuffer(data, dtype=np.uint8), (0, padded_length - length), mode='constant')

    img_matrix = data_padded.reshape(height, width)
    colored_img = Image.fromarray(img_matrix).convert('RGB')
    return colored_img
