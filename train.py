import torch
from src.dataset import License_Plate_Dataset
from src.d_model import D_Model
import argparse
import os
import random
import numpy as np
from src.g_model import G_Model
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm.autonotebook import tqdm
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', '-i', type=str, default='voc_plate_ocr_dataset/inputs')
    parser.add_argument('--target_dataset', '-t', type=str, default='voc_plate_ocr_dataset/targets')
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--learning_rate_G', '-l', type=float, default=0.001)
    parser.add_argument('--learning_rate_D', '-d', type=float, default=0.001)
    parser.add_argument('--image_size', '-s', type=int, default=112)
    parser.add_argument('--checkpoint_save', '-c', type=str, default='checkpoint')
    parser.add_argument('--D_checkpoint_load', '-k', type=str, default=None)
    parser.add_argument('--G_checkpoint_load', '-g', type=str, default=None)
    parser.add_argument('--tensorboard_dir', '-n', type=str, default='tensorboard')
    args = parser.parse_args()
    return args

def train(args):
    if not os.path.isdir(args.checkpoint_save):
        os.makedirs(args.checkpoint_save)

    if not os.path.isdir(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    # Train test split
    train_ratio = 0.8
    num_train = int(len(os.listdir(args.target_dataset)) * train_ratio)
    num_test = int(len(os.listdir(args.target_dataset)) - num_train)

    print(f'Number of train images: {num_train}')
    print(f'Number of valid images: {num_test}')

    random_seed = 42
    random.seed(random_seed)
    train_indexes = np.array(random.sample(range(num_test+num_train), num_train))
    mask = np.ones(num_train+num_test, dtype=bool)
    mask[train_indexes] = False

    inputs = [f"{i}.jpg" for i in range(num_train+num_test)]
    targets = [f"{i}.jpg" for i in range(num_train+num_test)]

    train_input_img_names = np.array(inputs)[train_indexes]
    train_target_img_names = np.array(targets)[train_indexes]
    test_input_img_names = np.array(inputs)[mask]
    test_target_img_names = np.array(targets)[mask]

    # train_input_img_paths = [os.path.join(args.input_dataset, file_name) for file_name in train_input_img_names]
    # train_target_img_paths = [os.path.join(args.target_dataset, file_name) for file_name in train_target_img_names]
    # test_input_img_paths = [os.path.join(args.input_dataset, file_name) for file_name in test_input_img_names]
    # test_target_img_paths = [os.path.join(args.target_dataset, file_name) for file_name in test_target_img_names]

    # Dataloader
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])
    train_dataset = License_Plate_Dataset(
        transform=transform,
        img_size=args.image_size, 
        input_image_path=train_input_img_names, 
        target_image_path=train_target_img_names,
        input_path=args.input_dataset,
        target_path=args.target_dataset,
    )

    test_dataset = License_Plate_Dataset(
        transform=transform,
        img_size=args.image_size, 
        input_image_path=test_input_img_names, 
        target_image_path=test_target_img_names,
        input_path=args.input_dataset,
        target_path=args.target_dataset,
    )

    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size= args.batch_size,
        shuffle= True,
        drop_last= True,
        num_workers= 0,
    )

    test_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size= args.batch_size,
        shuffle= False,
        drop_last= False,
        num_workers= 0,
    )

    print(len(train_dataloader))
    print(len(test_dataloader))

    # Load model
    writer = SummaryWriter(args.tensorboard_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = G_Model()
    discrimminator = D_Model()
    generator.to(device)
    discrimminator.to(device)

    d_optim = optim.SGD(params=discrimminator.parameters(), lr=args.learning_rate_D, momentum=0.9)
    g_optim = optim.SGD(params=generator.parameters(), lr=args.learning_rate_G, momentum=0.9)

    d_criteria = nn.BCEWithLogitsLoss()
    g_criteria = nn.L1Loss()

    # Continue training from checkpoint
    if args.D_checkpoint_load and args.G_checkpoint_load and os.path.isfile(args.D_checkpoint_load) and os.path.isfile(args.G_checkpoint_load):
        D_checkpoint = torch.load(args.D_checkpoint_load, weights_only=True)
        discrimminator.load_state_dict(D_checkpoint['model_params'])
        d_optim.load_state_dict(D_checkpoint['optimizer'])
        start_epoch = D_checkpoint['epoch'] + 1
        D_best_loss = D_checkpoint['best_loss']

        G_checkpoint = torch.load(args.G_checkpoint_load, weights_only=True)
        generator.load_state_dict(G_checkpoint['model_params'])
        g_optim.load_state_dict(G_checkpoint['optimizer'])
        start_epoch = G_checkpoint['epoch'] + 1
        G_best_loss = G_checkpoint['best_loss']
    else:
        start_epoch = 0
        G_best_loss = np.inf
        D_best_loss = np.inf

    num_iters = len(train_dataloader)
    # Train and validation
    for epoch in range(start_epoch, args.epoch):
        generator.train()
        discrimminator.train()

        progress_bar = tqdm(train_dataloader, colour='cyan')
        D_train_losses = []
        G_train_losses = []

        for iter, (inputs, true_images) in enumerate(progress_bar):
            # TRAIN D
            inputs = inputs.to(device)
            true_images = true_images.to(device)

            # G generate fake images 
            fake_images = generator(inputs)

            # D discrimminate and calculate fake images loss
            pred_fake = discrimminator(fake_images.to(device))
            loss_fake = d_criteria(pred_fake, torch.zeros(args.batch_size, device=device))

            # D discrimminate and calculate real images loss
            pred_real = discrimminator(true_images.to(device))
            loss_real = d_criteria(pred_real, torch.ones(args.batch_size, device=device))

            # Calculate average loss
            loss_D = (loss_fake+loss_real)/2.0
            writer.add_scalar('Train/D_Loss', loss_D.item(), epoch*num_iters+iter)
            D_train_losses.append(loss_D.item())

            # Backward
            d_optim.zero_grad()
            loss_D.backward()
            d_optim.step()

            # TRAIN G
            # Generate fake images
            fake_images = generator(inputs)

            # Calculate D loss
            pred_fake = discrimminator(fake_images.to(device))
            bce_loss_G = d_criteria(pred_fake, torch.ones_like(pred_fake, device=device))

            # Calculate F1 Loss
            l1_loss_G = g_criteria(fake_images, true_images)

            loss_G = bce_loss_G + l1_loss_G
            writer.add_scalar('Train/G_Loss', loss_D.item(), epoch*num_iters+iter)
            G_train_losses.append(loss_G.item())
            progress_bar.set_description(f'Train: Epoch {epoch+1}/{args.epoch}. G_Loss: {loss_G.item():0.4f}. D_Loss: {loss_D.item():0.4f}')

            # Backward
            g_optim.zero_grad()
            loss_G.backward()
            g_optim.step()

        # VALIDATION
        generator.eval()
        discrimminator.eval()

        G_valid_losses = []
        D_valid_losses = []

        for iter, (inputs, true_images) in enumerate(test_dataloader):
            with torch.inference_mode():
                inputs = inputs.to(device)
                true_images = true_images.to(device)

                fake_images = generator(inputs)
                pred_fake = discrimminator(fake_images.to(device))

                bce_loss_G = d_criteria(pred_fake, torch.ones_like(pred_fake, device=device))
                l1_loss_G = g_criteria(fake_images, true_images)
                loss_G = bce_loss_G + l1_loss_G

                loss_D = d_criteria(pred_fake, torch.zeros(args.batch_size, device=device))

                G_valid_losses.append(loss_G.item())
                D_valid_losses.append(loss_D.item())

        print(f'Val: Epoch {epoch+1}/{args.epoch}. G_Loss: {loss_G.item():0.4f}. D_Loss: {loss_D.item():0.4f}')
        G_valid_loss = np.mean(G_valid_losses)
        D_valid_loss = np.mean(D_valid_losses)
        
        writer.add_scalar('Valid/G_Loss', G_valid_loss, epoch)
        writer.add_scalar('Valid/D_Loss', D_valid_loss, epoch)

        # Save checkpoint
        G_checkpoint = {
            'epoch': epoch, 
            'model_params': generator.state_dict(),
            'optimizer': g_optim.state_dict(),
            'best_loss': G_best_loss,
        }

        D_checkpoint = {
            'epoch': epoch, 
            'model_params': discrimminator.state_dict(),
            'optimizer': d_optim.state_dict(),
            'best_loss': D_best_loss,
        }
        torch.save(G_checkpoint, os.path.join(args.checkpoint_save, 'G_last.pt'))
        torch.save(D_checkpoint, os.path.join(args.checkpoint_save, 'D_last.pt'))

        if D_valid_loss < D_best_loss:
            D_best_loss = D_valid_loss
            torch.save(discrimminator.state_dict(), os.path.join(args.checkpoint_save, 'D_best.pt'))
        
        if G_valid_loss < G_best_loss:
            G_best_loss = G_valid_loss
            torch.save(generator.state_dict(), os.path.join(args.checkpoint_save, 'G_best.pt'))



if __name__ == "__main__":
    args = get_args()
    train(args)