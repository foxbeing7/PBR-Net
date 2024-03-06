import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import numpy as np
from datasets import train_dataset, val_dataset
from attmodel import normal_UNet, albedo_UNet, pbr_unet, discriminator
from losses import normal_loss, albedo_loss, vgg_loss, tv_loss, D_loss ,G_loss
# from utils import compute_ssim
import os

parser = argparse.ArgumentParser(description='Aputure pbr network')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
parser.add_argument('--lr_step', type=int, default=10, help='lr schedule step')
parser.add_argument('--val_step', type=int, default=5, help='val step')
parser.add_argument('--size', type=int, default=512, help='size of input image')
parser.add_argument('--resume', type=str, default=None, help='load ckpts if resume is not none')

args = parser.parse_args()
# mask only works when vis is True
vis = False
mask = False
ckpt_path = args.resume


def main():
    learning_rate = args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal_unet_model = normal_UNet()
    albedo_unet_model = albedo_UNet()
    combined_unet_model = pbr_unet(normal_unet_model, albedo_unet_model).to(device)
    # d_model = discriminator().to(device)
    # todo
    if ckpt_path is not None:
        combined_unet_model.load_state_dict(torch.load(ckpt_path))

        # d_model.load_state_dict(torch.load('discriminator.pth'))

    optimizer = optim.Adam(combined_unet_model.parameters(), lr=learning_rate)
    # d_optimizer = optim.Adam(d_model.parameters(), lr=learning_rate)

    lr_scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                                               num_workers=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=2,
                                             shuffle=False)

    criterion_normal = normal_loss()
    criterion_albedo = albedo_loss()
    criterion_vgg = vgg_loss()
    # criterion_tv = tv_loss()
    # criterion_g = G_loss()
    # criterion_d = D_loss()

    best_combined_model_path = './ckpts/best_num180.pth'
    # best_d_model_path = './ckpts/best_d_model.pth'
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):

        train(epoch, combined_unet_model, train_loader, [optimizer],
              [criterion_normal, criterion_albedo, criterion_vgg], device)

        lr_scheduler.step()


        if epoch % args.val_step == 0:
            val_loss = validate(combined_unet_model, val_loader,
                                [criterion_normal, criterion_albedo, criterion_vgg], device)

            if val_loss < best_loss:
                best_loss = val_loss

                save_best_model(combined_unet_model, best_combined_model_path, epoch)
                # save_best_model(d_model,best_d_model_path,epoch)

                # predict test img when saved best model to visualize...jaja
                if vis:
                    output_folder = './output/testing'
                    normal_unet_model = normal_UNet()
                    albedo_unet_model = albedo_UNet()
                    combined_unet_model = pbr_unet(normal_unet_model, albedo_unet_model)
                    combined_unet_model.load_state_dict(torch.load(best_combined_model_path))
                    combined_unet_model.to(device)
                    combined_unet_model.eval()

                    input_image_folder = './dataset/test/test/image'
                    mask_image_folder = './dataset/test/test/mask'
                    for image_file in os.listdir(input_image_folder):
                        if image_file.endswith('.png') or image_file.endswith('.jpg'):
                            image_path = os.path.join(input_image_folder, image_file)
                            output_path = os.path.join(output_folder, image_file.split('.')[0])

                            # Construct mask path based on the image file name
                            mask_filename = f"{image_file.split('.')[0]}_mask.jpg"
                            mask_path = os.path.join(mask_image_folder, mask_filename)

                            # predict, pass mask_path as an argument
                            predict(combined_unet_model, image_path, output_path, mask_path, epoch)

    torch.save(combined_unet_model.state_dict(), "./ckpts/latest_num180.pth")
    # torch.save(d_model.state_dict(), "./ckpts/latest_d.pth")
    print(f'Saved latest model at path: ./ckpts/latest.pth')
    print('Training finished.')


def train(epoch, model,  train_loader, optimizers, criteria, device):
    model.train()
    # d_model.train()
    total_g_loss = 0.0
    # total_d_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['image'].to(device)
        targets_normal = batch['normal'].to(device)
        targets_albedo = batch['albedo'].to(device)

        # discriminator
        # optimizers[1].zero_grad()
        # _, fake_albedo = model(inputs)
        # d_targets_albedo = d_model(targets_albedo)
        # d_outputs_albedo = d_model(fake_albedo.detach())
        # loss_d = criteria[4](d_targets_albedo, d_outputs_albedo)
        # d_loss = loss_d
        # d_loss.backward()
        # optimizers[1].step()
        # total_d_loss += loss_d.item()


        # Generator
        optimizers[0].zero_grad()
        outputs_normal, outputs_albedo = model(inputs)
        # discriminator_fake_albedo = d_model(outputs_albedo)


        # loss
        loss_normal = criteria[0](outputs_normal, targets_normal)
        loss_albedo = criteria[1](outputs_albedo, targets_albedo)
        loss_vgg = criteria[2](outputs_albedo, targets_albedo)
        # loss_adv = criteria[3](discriminator_fake_albedo)
        # loss_tv = criteria[3](outputs_albedo)
        alpha = 1
        belta = 1
        gamma = 0.2
        # omiga = 0

        total_batch_loss = alpha * loss_normal + belta * loss_vgg + gamma * loss_albedo
        total_batch_loss.backward()
        optimizers[0].step()

        total_g_loss += total_batch_loss.item()



        print(
            f'Train Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, total_G_Loss: {total_batch_loss.item()},normal_loss:{alpha * loss_normal.item()}, '
            f'albedo_loss:{gamma * loss_albedo.item()}, vgg_loss: {belta * loss_vgg.item()}')

    avg_loss = total_g_loss / len(train_loader)
    print(f'Train Epoch {epoch}: Average Loss: {avg_loss}')
    return avg_loss


def validate(model, val_loader, criteria, device):
    model.eval()
    # d_model.eval()
    total_g_loss = 0.0
    # total_normal_ssim = 0.0
    # total_albedo_ssim = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch['image'].to(device)
            targets_normal = batch['normal'].to(device)
            targets_albedo = batch['albedo'].to(device)

            # Forward pass
            outputs_normal, outputs_albedo = model(inputs)
            # loss
            loss_normal = criteria[0](outputs_normal, targets_normal)
            loss_albedo = criteria[1](outputs_albedo, targets_albedo)
            loss_vgg = criteria[2](outputs_albedo, targets_albedo)

            # outputs_normal_np = outputs_normal.cpu().numpy()
            # targets_normal_np = targets_normal.cpu().numpy()
            # outputs_albedo_np = outputs_albedo.cpu().numpy()
            # targets_albedo_np = targets_albedo.cpu().numpy()
            #
            # ssim_normal = compute_ssim(outputs_normal_np,targets_normal_np)
            # ssim_albedo = compute_ssim(outputs_albedo_np,targets_albedo_np)
            # loss_tv = criteria[3](outputs_albedo)


            total_batch_loss = 0.2 * loss_albedo + loss_normal + loss_vgg

            total_g_loss += total_batch_loss.item()
            # total_normal_ssim += ssim_normal.item()
            # total_albedo_ssim += ssim_albedo.item()

    avg_val_loss = total_g_loss / len(val_loader)
    # avg_normal_ssim = total_normal_ssim / len(val_loader)
    # avg_albedo_ssim = total_albedo_ssim / len(val_loader)
    # print(f'Validation: Average Loss: {avg_val_loss}, Average normal SSIM: {avg_normal_ssim} , Average albedo SSIM: {avg_albedo_ssim}')
    print(f'Validation: Average Loss: {avg_val_loss}')
    return avg_val_loss


def save_best_model(model, path, epoch):
    torch.save(model.state_dict(), path)
    print(f'Saved best model at epoch:{epoch} with path: {path}')


def predict(model, input_path, output_path, mask_path, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    input_image = Image.open(input_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('L')

    mask_array = np.array(mask_image)
    binary_mask_array = (mask_array > 127.5).astype(np.uint8) * 255
    binary_mask_image = Image.fromarray(binary_mask_array)

    masked_image = Image.fromarray(np.array(input_image) * np.expand_dims(np.array(binary_mask_image) > 0, axis=2))
    resized_mask = mask_image.resize((args.size, args.size))
    input_tensor = transform(masked_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs_normal, outputs_albedo = model(input_tensor)

    os.makedirs(output_path, exist_ok=True)

    input_image.save(os.path.join(output_path, f'image_.png'))
    masked_image.save(os.path.join(output_path, f'masked_image_.png'))
    inv_normalize = transforms.Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225)
    )
    normal = inv_normalize(outputs_normal)
    normal_image = normal.squeeze(0).cpu().permute(1, 2, 0).numpy()
    resized_normal = Image.fromarray((normal_image * 255).astype('uint8')).resize((args.size, args.size))

    albedo = inv_normalize(outputs_albedo)
    albedo_image = albedo.squeeze(0).cpu().permute(1, 2, 0).numpy()
    resized_albedo = Image.fromarray((albedo_image * 255).astype('uint8')).resize((args.size, args.size))

    if mask:
        masked_normal = Image.fromarray(np.array(resized_normal) * np.expand_dims(np.array(resized_mask) > 0, axis=2))
        masked_normal.save(os.path.join(output_path, f'normal_{epoch}.png'))
        masked_albedo = Image.fromarray(np.array(resized_albedo) * np.expand_dims(np.array(resized_mask) > 0, axis=2))
        masked_albedo.save(os.path.join(output_path, f'albedo_{epoch}.png'))
    else:
        resized_normal.save(os.path.join(output_path, f'normal_{epoch}.png'))
        resized_albedo.save(os.path.join(output_path, f'albedo_{epoch}.png'))

    print(os.path.join("-------------------saved predicted result at : ", output_path, '------------------------'))


if __name__ == '__main__':
    main()
