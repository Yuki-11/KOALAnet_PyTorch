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

#### Coding now ...
class LocalConvDs(nn.module):
    def __init__(self, ch, k_sz):
        super(LocalConvDs, self).__init__()
        self.ch = ch
        self.k_sz = k_sz
        self.image_patches = ExtractSplitStackImagePatches(k_sz, k_sz)

    def forward(self, img, kernel_2d):
        # local filtering operation for features
        # img: [B, C, H, W]
        # kernel_2d: [B, kernel*kernel, H, W]
        img = self.image_patches(img) # [B, C*kh*kw, H, W]
        img = torch.split(img, self.k_sz**2, dim=1) # kh*kw of [B, C, H, W]
        img = torch.stack(img, dim=2) # [B, C, kh*kw, H, W]
    
        k_dim = kernel_2d.size()
        kernel_2d = kernel_2d.unsqueeze(1).expand(k_dim[0], self.ch, *k_dim[1:]).contiguous() # [B, C, kh*kw, H, W]

        y = torch.sum(img * kernel_2d, dim=2) # [B, C, kh*kw, H, W] -> [B, C, H, W]
        return y

## ============= UpwnsamplingNtowrk =============

class UpsamplingNetworkBaseline(nn.module):
    def __init__(self, in_ch, us_blur_k_sz, factor, out_ch=3):
        super(UpsamplingNetworkBaseline, self).__init__()
        self.md_ch = 64
        self.n_koala = 5
        self.n_res = 7

        self.conv1 = nn.Conv2d(in_ch, self.md_ch, kernel_size=3, padding=1)
        self.res_blocks_alt = nn.Sequential([ResBlock(self.md_ch, self.md_ch, 3) for i in range(self.n_koala)])
        self.res_blocks = nn.Sequential([ResBlock(self.md_ch, self.md_ch, 3) for i in range(self.n_res)])
        self.relu = nn.ReLU(inplace=True)
        self.upsampling_kernel_branch = nn.Sequential(
                                            nn.Conv2d(self.md_ch, self.md_ch * 2, 3, padding=1),
                                            nn.ReLU(inplace=False),
                                            nn.Conv2d(self.md_ch * 2, us_blur_k_sz ** 2 * factor ** 2, 3, padding=1),
                                            )
        img_branch_layers = []
        for i in range(int(np.log2(factor))):
            img_branch_layers += [nn.Conv2d(self.md_ch, self.md_ch * 2, 3, padding=1),
                                  nn.ReLU(inplace=False),
                                  nn.PixelShuffle(2),
                                  ]
        self.rgb_res_img_branch = nn.Sequential(*img_branch_layers,
                                                nn.Conv2d(self.md_ch, out_ch, 3, padding=1),
                                                )
        self.local_conv_us = LocalConvUs()

    def forward(self, x, k2d_ds, factor, kernel, channels=3):
        # extract degradation kernel features
        filter_koala_list = []
        h = self.conv1(x)
        h = self.res_blocks_alt(h)
        h = self.relu(self.res_blocks(h))
        # upsampling kernel branch
        k2d = self.upsampling_kernel_branch(h)
        # rgb residual image branch
        rgb = self.rgb_res_img_branch(h)
        # local filtering and upsampling
        output_k2d = self.local_conv_us(k2d)
        output = output_k2d + rgb

        return output


class UpsamplingNetwork(UpsamplingNetworkBaseline):
    def __init__(self, in_ch, ds_blur_k_sz, lc_blur_k_sz, us_blur_k_sz, factor, out_ch=3):
        super(UpsamplingNetwork, self).__init__(in_ch, us_blur_k_sz, factor, out_ch)
        self.cor_ker_block = CorrectKernelBlock(ds_blur_k_sz**2, self.md_ch, 3)
        self.koala_modules = nn.Sequential([KOALAModlule(self.md_ch, self.md_ch, covn_k_sz=3, lc_k_sz=lc_blur_k_sz) for i in range(self.n_koala)])

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

        return output, filter_koala_list[-1]


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
    def __init__(self, ch, k_sz, factor):
        super(LocalConvUs, self).__init__()
        self.ch = ch
        self.k_sz = k_sz
        self.image_patches = ExtractSplitStackImagePatches(k_sz, k_sz)
        self.factor = factor
        self.kernel_norm = KernelNormalize(k_sz)
        self.pixel_shuffle = nn.PixelShuffle(factor)

    def forward(self, img, kernel_2d):
        # local filtering operation for upsampling network
        # img: [B, C, H, W]
        # kernel_2d: [B, k_sz*k_sz*factor*factor, H, W]
        img = self.image_patches(img) # [B, C*kh*kw, H, W]
        img = torch.split(img, self.k_sz**2, dim=1) # kh*kw of [B, C, H, W]
        img = torch.stack(img, dim=2) # [B, C, kh*kw, H, W]
        img_dim = img.size()
        img_dim[1] *= self.factor * 2
        img = img.expand(*img_dim).contiguous() # [B, C*factor**2, kh*kw, H, W]

        kernel_2d = torch.split(kernel_2d, self.k_sz**2, dim=1) # kh*kw of [B, factor**2, H, W]
        kernel_2d = torch.stack(kernel_2d, dim=2) # [B, factor**2, kh*kw, H, W]
        kernel_2d = self.kernel_norm(kernel_2d, dim=2) # [B, factor**2, kh*kw, H, W]
        k_dim = kernel_2d.size()
        kernel_2d = kernel_2d.unsqueeze(1).expand(k_dim[0], self.ch, *k_dim[1:]).contiguous() # [B, C, factor**2, kh*kw, H, W]
        kernel_2d = torch.unbind(kernel_2d, dim=2) # factor**2 of [B, C, kh*kw, H, W]
        kernel_2d = torch.cat(kernel_2d, dim=1) # [B, C*factor**2, kh*kw, H, W]

        result = torch.sum(img * kernel_2d, dim=2) # [B, C*factor**2, kh*kw, H, W] -> [B, C*factor**2, H, W]
        result = self.pixel_shuffle(result) # [B, C*factor**2, H, W] -> [B, C, H*factor, W*factor]
        return result


class LocalConvFeat(nn.module):
    def __init__(self, ch, k_sz):
        super(LocalConvFeat, self).__init__()
        self.ch = ch
        self.k_sz = k_sz
        self.image_patches = ExtractSplitStackImagePatches(k_sz, k_sz)

    def forward(self, img, kernel_2d):
        # local filtering operation for features
        # img: [B, C, H, W]
        # kernel_2d: [B, kernel*kernel, H, W]
        img = self.image_patches(img) # [B, C*kh*kw, H, W]
        img = torch.split(img, self.k_sz**2, dim=1) # kh*kw of [B, C, H, W]
        img = torch.stack(img, dim=2) # [B, C, kh*kw, H, W]
    
        k_dim = kernel_2d.size()
        kernel_2d = kernel_2d.unsqueeze(1).expand(k_dim[0], self.ch, *k_dim[1:]).contiguous() # [B, C, kh*kw, H, W]

        y = torch.sum(img * kernel_2d, dim=2) # [B, C, kh*kw, H, W] -> [B, C, H, W]
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
        patches = patches.permute(0, 1, 4, 5, 2, 3).contiguous() # [B, C, kh, kw, H, W]
        patches = patches.view(*patches.size()[0], -1, *patches.size()[4:]) # [B, C*kh*kw, H, W]
        return patches


class KernelNormalize(nn.module):
    def __init__(self, k_sz):
        super(KernelNormalize, self).__init__()
        self.k_sz = k_sz

    def forward(self, kernel_2d, dim=1):
        kernel_2d = kernel_2d - torch.mean(kernel_2d, dim, True)
        kernel_2d = kernel_2d + 1.0 / (self.k_sz ** 2)
        return kernel_2d


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


# ================= CommonModule ======================

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




