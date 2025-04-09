import numpy as np
from PIL import Image
import math

def pe_to_image(pe_file, width=256):
    with open(pe_file, 'rb') as f:
        content = f.read()

    binary_values = np.frombuffer(content, dtype=np.uint8)
    height = math.ceil(len(binary_values) / width)
    padded_length = width * height
    padded_binary = np.pad(binary_values, (0, padded_length - len(binary_values)), 'constant', constant_values=0)

    image_array = padded_binary.reshape(height, width)
    image = Image.fromarray(image_array).convert('RGB')
    
    return image
