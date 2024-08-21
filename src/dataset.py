import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

class License_Plate_Dataset(Dataset):
    def __init__(self, transform, img_size, input_image_path, target_image_path, input_path='voc_plate_ocr_dataset/inputs', target_path='voc_plate_ocr_dataset/targets') -> None:
        self.transform = transform
        self.img_size = img_size

        self.input_images = [os.path.join(input_path, file) for file in input_image_path]
        self.target_images = [os.path.join(target_path, file) for file in target_image_path]


    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        input_path = self.input_images[index]
        target_path = self.target_images[index]

        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
        Resize((112, 112))
    ])
    dataset = License_Plate_Dataset(img_size=112, input_image_path=['0.jpg'], target_image_path=['1.jpg'], transform=transform)
    image_1, image_2 = dataset[0]
    # print(image_1.shape)
    # print(image_2.shape)