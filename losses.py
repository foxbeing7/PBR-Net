import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import torch
import torchvision.models as models
import torchvision

#
class normal_loss(nn.Module):
    def __init__(self):
        super(normal_loss,self).__init__()

    def forward(self, predicted_normal, target_normal):
        smooth_l1 = F.smooth_l1_loss(predicted_normal,target_normal)
        return smooth_l1

class albedo_loss(nn.Module):
    def __init__(self):
        super(albedo_loss,self).__init__()

    def forward(self, predicted_normal, target_normal):
        smooth_l1 = F.smooth_l1_loss(predicted_normal,target_normal)
        return smooth_l1

class vgg_loss(nn.Module):
    def __init__(self, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg.to(device)
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss

class tv_loss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(tv_loss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class D_loss(nn.Module):
    def __init__(self):
        super(D_loss,self).__init__()

    def forward(self, real_discriminator_output, fake_discriminator_output):
        real_loss = F.binary_cross_entropy_with_logits(real_discriminator_output, torch.zeros_like(real_discriminator_output))
        fake_loss = F.binary_cross_entropy_with_logits(fake_discriminator_output, torch.ones_like(fake_discriminator_output))

        d_loss = 0.5 * (real_loss + fake_loss)

        return d_loss

class G_loss(nn.Module):
    def __init__(self):
        super(G_loss,self).__init__()

    def forward(self,generated_data):
        adversarial_loss = F.binary_cross_entropy_with_logits(generated_data,torch.ones_like(generated_data))

        return adversarial_loss

