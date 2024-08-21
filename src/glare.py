from PIL import Image, ImageEnhance, ImageOps
import random
import os
import numpy as np

def apply_sun_effect(img):
    image = Image.open(img)
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(2.8, 4.5)
    return enhancer.enhance(factor)

def apply_gaussian_noise(img):
    img_np = np.array(img)

    # Generate Gaussian noise
    noise = np.random.normal(loc=0.0, scale=25, size=img_np.shape)
    noisy_img = img_np + noise  # Add noise to the image
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img)

def add_glare_effect(img):
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))

    gradient = ImageOps.colorize(
        ImageOps.grayscale(Image.new("RGB", img.size)), black="black", white="white"
    )
    overlay = Image.blend(overlay, gradient.convert("RGBA"), 0.5)
    img = Image.blend(img.convert("RGBA"), overlay, 0.3)
    return img.convert("RGB")

if __name__ == "__main__":
    input_path = "voc_plate_ocr_dataset/targets"
    output_path = "voc_plate_ocr_dataset/inputs"

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for image_path in os.listdir(input_path):
        file_name = os.path.basename(image_path.split('.')[0])
        file_name += '.jpg'

        modified_image = apply_sun_effect(os.path.join(input_path, image_path))

        modified_image = add_glare_effect(modified_image)

        modified_image.save(os.path.join(output_path, file_name))


