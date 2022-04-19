import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models
from torch.autograd import Variable

class PerceptualLoss():
    def initialize(self):
        conv_layer = 14
        cnn = models.vgg19(pretrained=True).features.cuda()
        self.model = nn.Sequential().cuda()
        for i, layer in enumerate(list(cnn)):
            self.model.add_module(str(i), layer)
            if i == conv_layer:
                break

    def get_loss(self, fake, real):
        f_fake = self.model.forward(fake)
        f_real = self.model.forward(real).detach()
        self.criterion = nn.L1Loss()
        loss = self.criterion(f_fake, f_real)
        return loss

class WGANLoss():
    def compute_pixel_loss(self, net, real, fake):
        mse_loss = nn.MSELoss()
        self.pixel_loss = mse_loss(real, fake)
        return self.pixel_loss
    def compute_gradient_penalty(self, real, fake, net_D):
        self.Lambda = 10
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real.size()).cuda()
        compose = alpha * real + (1 - alpha) * fake
        compose = Variable(compose.cuda(), requires_grad=True)
        D_compose = net_D.forward(compose)
        gradients = autograd.grad(outputs=D_compose, inputs=compose, grad_outputs=torch.ones(D_compose.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)
        gradients_penalty = ((gradients[0].norm(2, dim=1) - 1) ** 2).mean() * self.Lambda
        return gradients_penalty
    def compute_RaLSGAN_loss(self, input_fake, input_real, net_D):
        input_fake = input_fake.detach()
        input_real = input_real.detach()
        output_fake = net_D.forward(input_fake)
        output_real = net_D.forward(input_real)
        Ra_LSGAN_loss = (torch.mean((output_real - torch.mean(output_fake) - 1) ** 2) +
                torch.mean((output_fake - torch.mean(output_real) - 1) ** 2)) / 2
        return Ra_LSGAN_loss