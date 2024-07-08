import glob
import os

images = glob.glob(r"C:\Users\zutom\.vscode\PythonFile\ir_image_frame*.png")
print(images)

for enum, image in enumerate(images):
    print(f"image = {image}")
    os.remove(image)