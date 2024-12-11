import os
from PIL import Image
from tqdm import tqdm

def make_gif(input_folder, output_file, duration=500):
    images = []
    for file_name in tqdm(sorted(os.listdir(input_folder))):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, file_name)
            img = Image.open(file_path)
            img = img.resize((1080, 720), Image.ANTIALIAS)  # Resize to HD resolution
            images.append(img)

    if images:
        images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=0)

input_folder = r'G:\gait_pattern\int_cali\tkrzk_9g\Intrinsic_fr'
output_file = r'G:\gait_pattern\int_cali\tkrzk_9g\Intrinsic_fr.gif'
make_gif(input_folder, output_file)