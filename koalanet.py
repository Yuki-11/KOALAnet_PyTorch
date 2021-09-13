import torch
import torchvision
import torch.nn.functional as F
from torch import nn
import numpy as np


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


## ============= KOALAnet =============


class KOALAnet(nn.module):
    def __init__(self):
        super(KOALAnet, self).__init__()

    def forward(self, x):
        return x


## ============= UpwnsamplingNtowrk =============

class UpsamplingNetwork(nn.module):
    def __init__(self, in_ch, ds_blur_k_sz, us_blur_k_sz, factor, out_ch=3):
        super(UpsamplingNetwork, self).__init__()
        md_ch = 64
        n_koala = 5
        n_res = 7
        
        self.cor_ker_block = CorrectKernelBlock(ds_blur_k_sz**2, md_ch, 3)

        self.conv1 = nn.Conv2d(in_ch, md_ch, kernel_size=3, padding=1)
        self.koala_modules = nn.Sequential([KOALAModlule(md_ch, md_ch, covn_k_sz=3, lc_k_sz=7) for i in range(n_koala)])
        self.res_blocks = nn.Sequential([ResBlock(md_ch, md_ch, 3) for i in range(n_res)])
        self.relu = nn.ReLU(inplace=True)
        self.upsampling_kernel_branch = nn.Sequential(
                                            nn.Conv2d(md_ch, md_ch * 2, 3, padding=1),
                                            nn.ReLU(inplace=False),
                                            nn.Conv2d(md_ch * 2, us_blur_k_sz ** 2 * factor ** 2, 3, padding=1),
                                            )
        img_branch_layers = []
        for i in range(int(np.log2(factor))):
            img_branch_layers += [nn.Conv2d(md_ch, md_ch * 2, 3, padding=1),
                                  nn.ReLU(inplace=False),
                                  nn.PixelShuffle(2),
                                  ]
        self.rgb_res_img_branch = nn.Sequential(*img_branch_layers,
                                                nn.Conv2d(md_ch, out_ch, 3, padding=1),
                                                )
        self.local_conv_us = LocalConvUs()

    def forward(self, x, k2d_ds, factor, kernel, channels=3):
        # extract degradation kernel features
        filter_koala_list = []
        h = self.conv1(x)
        k = self.cor_ker_block(k2d_ds)
        h, filter_koala_list = self.koala_modules(h, k, filter_koala_list)
        h = self.relu(self.res_blocks(h))
        # upsampling kernel branch
        k2d = self.upsampling_kernel_branch(h)
        # rgb residual image branch
        rgb = self.rgb_res_img_branch(h)
        # local filtering and upsampling
        output_k2d = self.local_conv_us(k2d)
        output = output_k2d + rgb

        return output


class KOALAModlule(nn.module):
    def __init__(self, in_ch, out_ch, covn_k_sz=3, lc_k_sz=7):
        super(KOALAModlule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=covn_k_sz, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(out_ch, out_ch, kernel_size=covn_k_sz, padding=1),
                                nn.ReLU(inplace=True),
                                )
        self.mult_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=covn_k_sz, padding=1),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(out_ch, out_ch, kernel_size=covn_k_sz, padding=1),
                                    )
        self.loc_filter_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=1),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(in_ch, lc_k_sz**2, kernel_size=1, padding=1),
                                    KernelNormalize(lc_k_sz)
                                    )
        self.local_conv = LocalConvFeat(out_ch, lc_k_sz)

    def forward(self, x, kernel, filter_koala_list):
        h = self.conv(x)
        m = self.mult_conv(kernel)
        h = h * m
        k = self.loc_filter_conv(kernel)
        h = self.local_conv(h, k)

        return x+h, filter_koala_list.append(k)


class CorrectKernelBlock(nn.module):
    def __init__(self, in_ch, out_ch, num_blocks, k_sz=3):
        super(CorrectKernelBlock, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k_sz, padding=1),
                nn.ReLU(inplace=True)]
        for i in range(num_blocks-1):
            layers += [nn.Conv2d(out_ch, out_ch, kernel_size=k_sz, padding=1),
                        nn.ReLU(inplace=True)]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class LocalConvUs(nn.module):
    def __init__(self, ch, k_sz):
        super(LocalConvUs, self).__init__()
        self.ch = ch
        self.k_sz = k_sz
        self.image_patches = ExtractSplitStackImagePatches(k_sz, k_sz)

    def forward(self, img, kernel_2d):
        kernel_2d = kernel_2d.expand(-1, self.ch).permute(0, 3, 1, 2).contiguous()
        img = self.image_patches(img)
        y = torch.sum(img * kernel_2d, dim=2) # [B, C, kh*hw, H, W] -> [B, C, H, W]
        return y


## ============= DownsamplingNtowrk =============

class DownsamplingNetwork(nn.module):
    def __init__(self, in_ch, blur_k_sz, n_res):
        super(DownsamplingNetwork, self).__init__()
        md_ch = 64
        conv_k_sz = 3

        self.enc1 = EncBlock(in_ch, md_ch, n_res),
        self.enc2 = EncBlock(md_ch, md_ch*2, n_res)

        self.Bottlenec = BottlenecResBlcok(md_ch*2, md_ch*4)

        self.dec1 = DecBlock(md_ch*4, md_ch*2, n_res)
        self.dec2 = DecBlock(md_ch*2, md_ch, n_res)
        self.dec3 = nn.Sequential(nn.Conv2d(md_ch, md_ch, kernel_size=conv_k_sz, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(md_ch, blur_k_sz ** 2, kernel_size=conv_k_sz, padding=1)
                                )

    def forward(self, x):
        # encoder
        skips = {}
        h, skips[0] = self.enc1(x)
        h, skips[1] = self.enc2(h)
        # bottleneck
        h = self.Bottlenec(h)
        # decoder
        h = self.dec1(h, skips[0])
        h = self.dec1(h, skips[1])
        # downsampling kernel branch
        k2d = self.dec3(h)

        return k2d


class EncBlock(nn.module):
    def __init__(self, in_ch, out_ch, n_res):
        super(EncBlock, self).__init__()
        k_sz = 3
        res_blocks = [ResBlock(out_ch, out_ch, k_sz) for i in range(n_res)]
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                *res_blocks,
                nn.ReLU(inplace=True),
                ]
        self.encode = nn.Sequential(*layers)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        skip = self.encode(x)
        h = self.max_pool(skip)
        return h, skip


class BottlenecResBlcok(nn.module):
    def __init__(self, in_ch, out_ch):
        k_sz = 3
        super(BottlenecResBlcok, self).__init__()
        self.f = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                ResBlock(out_ch, out_ch, k_sz),
                                nn.ReLU(inplace=True),
                                )
    def forward(self, x):
        return self.f(x)


class DecBlock(nn.module):
    def __init__(self, in_ch, out_ch, n_res):
        super(DecBlock, self).__init__()
        k_sz = 4
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, k_sz, padding=2) # Scale factor x2
        res_blocks = [ResBlock(out_ch, out_ch, k_sz) for i in range(n_res)]
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                *res_blocks,
                nn.ReLU(inplace=True),
                ]
        self.conv_res = nn.Sequential(*layers)

    def forward(self, x, skip):
        h = self.deconv(x)
        h = torch.cat((h, skip), 0)
        y = self.conv_res(h)
        return y


class ResBlock(nn.module):
    def __init__(self, in_ch, out_ch, k_sz):
        super(ResBlock, self).__init__()
        self.f = nn.Sequential(nn.ReLU(inplace=True),
                                nn.Conv2d(in_ch, out_ch, kernel_size=k_sz, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(out_ch, out_ch, kernel_size=k_sz, padding=1)
                                )

    def forward(self, x):
        y = self.f(x)
        return y+x


class KernelNormalize(nn.module):
    def __init__(self, k_sz):
        super(KernelNormalize, self).__init__()
        self.k_sz = k_sz

    def forward(self, kernel_2d):
        kernel_2d = kernel_2d - torch.mean(kernel_2d, 3, True)
        kernel_2d = kernel_2d + 1.0 / (self.k_sz ** 2)
        return kernel_2d


class LocalConvFeat(nn.module):
    def __init__(self, ch, k_sz):
        super(LocalConvFeat, self).__init__()
        self.ch = ch
        self.k_sz = k_sz
        self.image_patches = ExtractSplitStackImagePatches(k_sz, k_sz)

    def forward(self, img, kernel_2d):
        kernel_2d = kernel_2d.expand(-1, self.ch).permute(0, 3, 1, 2).contiguous()
        img = self.image_patches(img)
        y = torch.sum(img * kernel_2d, dim=2) # [B, C, kh*hw, H, W] -> [B, C, H, W]
        return y


class ExtractSplitStackImagePatches(nn.module):
    def __init__(self, kh, kw, padding="same"):
        super(ExtractSplitStackImagePatches, self).__init__()
        # stride = 1
        self.k_sz = [kh, kw]
        if padding == 'same':
            self.pad = [(kw - 1)/2 for i in range(2)] + [(kh - 1)/2 for i in range(2)]
        else:
            self.pad = [0, 0]
        self.stride = [1, 1]

    def forward(self, x):
        # https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/8
        x = F.pad(x, self.pad)
        patches = x.unfold(2, self.k_sz[0], self.stride[0]).unfold(3, self.k_sz[1], self.stride[1])
        patches = patches.permute(0, 1, 4, 5, 2, 3).contiguous() # [B, C, kh, hw, H, W]
        patches = patches.view(*patches.size()[:2], -1, *patches.size()[4:]) # [B, C, kh*hw, H, W]
        return patches

