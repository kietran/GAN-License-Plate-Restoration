from PIL import Image
import numpy as np
import os
from src.g_model import G_Model
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Compose
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default=None, help='Path to image or directory')
    parser.add_argument('--g_checkpoint', '-g', type=str, default='checkpoint/G_best.pt')
    parser.add_argument('--img_size', '-s', type=int, default=112)
    args = parser.parse_args()
    return args

def inference(args):
    checkpoint = args.g_checkpoint
    img_size = args.img_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_transform(image_path):
        try:
            image = Image.open(image_path)
        except:
            print(f'Failed to load image: {image_path}')
            return None
        transform = Compose([
            ToTensor(),
            Resize((img_size, img_size))
        ])
        return transform(image).to(device)

    # G Model
    generator = G_Model()
    generator.to(device)
    generator.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    generator.eval()
    
    
    if os.path.isfile(args.input_path):
        # Single image inference
        image = load_and_transform(args.input_path)
        if image is None:
            exit(1)

        fig, axes = plt.subplots(1, 2, figsize=(5, 5))
        axes[0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
        axes[0].set_title('Input')
        axes[0].axis('off')

        generated_image = generator(image.unsqueeze(0)).detach().cpu().numpy()[0]
        axes[1].imshow(np.transpose(generated_image, (1, 2, 0)))
        axes[1].set_title('Generated')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('Test.jpg')
        plt.show()

    elif os.path.isdir(args.input_path):
        # Multiple images inference
        input_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.jpg')]
        num_images = len(input_files)

        if num_images == 0:
            print("No images found in the directory.")
            exit(1)

        print(f'Number of test images: {num_images}')

        fig, axes = plt.subplots(num_images, 2, figsize=(15, num_images * 8))

        for i, img_path in enumerate(input_files):
            image = load_and_transform(img_path)
            if image is None:
                continue

            # Display input image
            axes[i, 0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')

            # Generate and display output image
            generated_image = generator(image.unsqueeze(0)).detach().cpu().numpy()[0]
            axes[i, 1].imshow(np.transpose(generated_image, (1, 2, 0)))
            axes[i, 1].set_title('Generated')
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig('Testt.jpg')
        plt.show()

    else:
        print("Input path is neither a file nor a folder.")
        exit(1)

if __name__ == "__main__":
    args = get_args()
    inference(args)