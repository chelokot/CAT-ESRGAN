import ntpath
import os

import numpy as np
import torch
from torch import nn

import yaml
from torch.cuda import amp
from tqdm import tqdm

from data import create_eval_dataloader
from utils import util
from .base_model import BaseModel
from .esrgan.imgproc import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, \
    random_horizontally_flip_torch

from esrgan.train_gan import build_model, define_loss, define_optimizer


class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert is_train
        parser = super(Pix2PixModel, Pix2PixModel).modify_commandline_options(
            parser, is_train)
        parser.add_argument('--config_path', type=str, required=True, help='path to config file')
        # parser.add_argument(
        #     '--real_stat_path',
        #     type=str,
        #     required=True,
        #     help=
        #     'the path to load the groud-truth images information to compute FID.'
        # ) #TODO: what is this?
        return parser

    def __init__(self, opt):
        # """Initialize the esrgan class.
        #
        # Parameters:
        #     opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        # """
        assert opt.isTrain
        BaseModel.__init__(self, opt)

        self.loss_names = [
            'G_pixel', 'G_content', 'G_adversarial', 'G_total', 'D_gt', 'D_sr', 'D_total'
        ]
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D']

        with open(opt.config_path, "r") as f:
            self.config = yaml.full_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG, self.netGema, self.netD = build_model(self.config, self.device)

        self.pixel_weight = torch.Tensor(self.config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"]).to(self.device)
        self.content_weight = torch.Tensor(self.config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["WEIGHT"]).to(self.device)
        self.adversarial_weight = torch.Tensor(self.config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["WEIGHT"]).to(self.device)

        self.pixel_criterion, self.content_criterion, self.adversarial_criterion = define_loss(self.config, self.device)
        self.optimizer_G, self.optimizer_D = define_optimizer(self.netG, self.netD, self.config)

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader = create_eval_dataloader(self.opt)
        self.best_psnr = 1e9
        self.psnrs = []
        self.is_best = False
        self.Tacts, self.Sacts = {}, {}
        # self.npz = np.load(opt.real_stat_path)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.real_B, self.real_A = random_crop_torch(self.real_B,
                                   self.real_A,
                                   self.config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"],
                                   self.config["SCALE"])
        self.real_B, self.real_A = random_rotate_torch(self.real_B, self.real_A, self.config["SCALE"], [0, 90, 180, 270])
        self.real_B, self.real_A = random_vertically_flip_torch(self.real_B, self.real_A)
        self.real_B, self.real_A = random_horizontally_flip_torch(self.real_B, self.real_A)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            gt_output = self.netD(self.real_B)
            sr_output = self.netD(self.fake_B.detach().clone())
            d_loss_gt = self.adversarial_criterion(gt_output - torch.mean(sr_output), torch.ones_like(gt_output, device = self.device)) * 0.5
        d_loss_gt.backward(retain_graph=True)

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            sr_output = self.netD(self.fake_B.detach().clone())
            d_loss_sr = self.adversarial_criterion(sr_output - torch.mean(gt_output), torch.zeros_like(gt_output, device = self.device)) * 0.5
        d_loss_sr.backward()

        # Calculate the total discriminator loss value
        d_loss = d_loss_gt + d_loss_sr

        self.loss_D_gt = d_loss_gt.item()
        self.loss_D_sr = d_loss_sr.item()
        self.loss_D_total = d_loss.item()

    def backward_G(self):
        # Initialize the generator model gradient
        self.netG.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
        with amp.autocast():
            sr = self.netG(self.real_A)
            # Output discriminator to discriminate object probability
            gt_output = self.netD(self.real_B.detach().clone())
            sr_output = self.netD(sr)
            pixel_loss = self.pixel_criterion(sr, self.real_B)
            content_loss = self.content_criterion(sr, self.real_B)
            d_loss_gt = self.adversarial_criterion(gt_output - torch.mean(sr_output), torch.zeros_like(gt_output, device = self.device)) * 0.5
            d_loss_sr = self.adversarial_criterion(sr_output - torch.mean(gt_output), torch.ones_like(gt_output, device = self.device)) * 0.5
            adversarial_loss = d_loss_gt + d_loss_sr
            pixel_loss = torch.sum(torch.mul(self.pixel_weight, pixel_loss))
            content_loss = torch.sum(torch.mul(self.content_weight, content_loss))
            adversarial_loss = torch.sum(torch.mul(self.adversarial_weight, adversarial_loss))
            # Compute generator total loss
            g_loss = pixel_loss + content_loss + adversarial_loss
        # Backpropagation generator loss on generated samples
        g_loss.backward()

        self.loss_G_pixel = pixel_loss.item()
        self.loss_G_content = content_loss.item()
        self.loss_G_adversarial = adversarial_loss.item()
        self.loss_G_total = g_loss.item()

    def optimize_parameters(self, steps):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def evaluate_model(self, step, save_image=False):
        self.is_best = False

        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG.eval()

        fakes, names = [], []
        cnt = 0
        psnr = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader)):
            self.set_input(data_i)
            self.test()
            mse = torch.mean((self.fake_B - self.real_B) ** 2)
            psnr += 10 * torch.log10(1 / mse)
            fakes.append(self.fake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10 or save_image:
                    input_im = util.tensor2im(self.real_A[j])
                    real_im = util.tensor2im(self.real_B[j])
                    fake_im = util.tensor2im(self.fake_B[j])
                    util.save_image(input_im,
                                    os.path.join(save_dir, 'input',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(real_im,
                                    os.path.join(save_dir, 'real',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(fake_im,
                                    os.path.join(save_dir, 'fake',
                                                 '%s.png' % name),
                                    create_dir=True)
                cnt += 1

        psnr /= len(self.eval_dataloader)
        if psnr < self.best_psnr:
            self.is_best = True
            self.best_psnr = psnr
        self.psnrs.append(psnr)
        if len(self.psnrs) > 3:
            self.psnrs.pop(0)

        ret = {
            'metric/psnr': psnr,
            'metric/psnr-mean': sum(self.psnrs) / len(self.psnrs),
            'metric/psnr-best': self.best_psnr
        }

        self.netG.train()
        return ret
