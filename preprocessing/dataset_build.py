import os
from tqdm import tqdm
from pe_to_image import pe_to_image

def build_dataset(pe_dir, image_dir, width=256):
    os.makedirs(image_dir, exist_ok=True)
    for file in tqdm(os.listdir(pe_dir)):
        img = pe_to_image(os.path.join(pe_dir, file), width)
        img.save(os.path.join(image_dir, f"{os.path.splitext(file)[0]}.png"))
