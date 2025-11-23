# from .base import Misfit
# import torch
# import torch.nn.functional as F
      
# class Misfit_M_SSIM(Misfit):
#     def __init__(self, gaussian_sigmas=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
#                  data_range = 1.0,
#                  K=(0.01, 0.03),
#                  alpha=1,
#                  compensation=200.0):
#         super(Misfit_M_SSIM, self).__init__()
#         self.C1 = (K[0] * data_range) ** 2
#         self.C2 = (K[1] * data_range) ** 2
#         self.pad = int(4 * gaussian_sigmas[-1])
#         self.alpha = alpha
#         self.compensation=compensation
#         filter_size = int(8 * gaussian_sigmas[-1] + 1)
        
#         # Create the Gaussian masks for convolution
#         g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
#         for idx, sigma in enumerate(gaussian_sigmas):
#             g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        
#         self.g_masks = g_masks

#     def _fspecial_gauss_1d(self, size, sigma):
#         """Create 1-D gauss kernel"""
#         coords = torch.arange(size).to(dtype=torch.float)
#         coords -= size // 2
#         g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#         g /= g.sum()
#         return g.reshape(-1)

#     def _fspecial_gauss_2d(self, size, sigma):
#         """Create 2-D gauss kernel"""
#         gaussian_vec = self._fspecial_gauss_1d(size, sigma)
#         return torch.outer(gaussian_vec, gaussian_vec)
#     def mean_filter(self,x, window_size=3):
#     	kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
#     	kernel = kernel.to(x.device)
#     	return F.conv2d(x, kernel, padding=window_size//2)

#     def forward(self, obs, syn):

#         syn = syn/(1e-10 + torch.max(torch.abs(syn),axis=1,keepdim=True).values)
#         obs = obs/(1e-10 + torch.max(torch.abs(obs),axis=1,keepdim=True).values)
#         # max_val = torch.max(torch.abs(syn), dim=1, keepdim=True).values
#         # max_val = torch.max(max_val, dim=2, keepdim=True).values
#         # syn = syn/(1e-20 + max_val)
#         # max_val = torch.max(torch.abs(obs), dim=1, keepdim=True).values
#         # max_val = torch.max(max_val, dim=2, keepdim=True).values
#         # obs = obs/(1e-20 + max_val)
#         obs = obs.unsqueeze(1)  # (B, 1, T, R)
#         syn = syn.unsqueeze(1)  # (B, 1, T, R)
        
      
#         self.g_masks = self.g_masks.to(syn.device)
               
#         # Apply 2D convolution with Gaussian kernels to get the necessary statistics
#         mux = F.conv2d(obs, self.g_masks, groups=1, padding=self.pad)
#         muy = F.conv2d(syn, self.g_masks, groups=1, padding=self.pad)

#         mux2 = mux * mux
#         muy2 = muy * muy
#         muxy = mux * muy

#         sigmax2 = F.conv2d(obs * obs, self.g_masks, groups=1, padding=self.pad) - mux2
#         sigmay2 = F.conv2d(syn * syn, self.g_masks, groups=1, padding=self.pad) - muy2
#         sigmaxy = F.conv2d(obs * syn, self.g_masks, groups=1, padding=self.pad) - muxy

#         # l, cs in MS-SSIM
#         L  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 1, T, R]
#         CS = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

#         # MS-SSIM loss calculation       
#         loss_ms_ssim = torch.mean(-(L*CS))
            

#         # loss
#         loss_mix = self.alpha * loss_ms_ssim
#         loss_mix = self.compensation * loss_mix

#         return loss_mix.mean()+200


from .base import Misfit
import torch
import torch.nn.functional as F
import math

class Misfit_M_SSIM(Misfit):
    def __init__(self, gaussian_sigmas=[0.75, 1.0, 1.25, 1.5, 2.0],
                 data_range=1.0,
                 K=(0.01, 0.03),
                 alpha=1.0,
                 compensation=200.0,
                 is_3d=False):

        super().__init__()
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.alpha = alpha
        self.compensation = compensation
        self.gaussian_sigmas = gaussian_sigmas
        self.is_3d = is_3d

        self._build_kernels(is_3d)

    def _build_kernels(self, is_3d):
        # 根据维度初始化高斯核
        max_sigma = self.gaussian_sigmas[-1]
        self.pad = int(4 * max_sigma)
        filter_size = int(8 * max_sigma + 1)

        if is_3d:
            # 3D 高斯核
            g_masks = torch.zeros((len(self.gaussian_sigmas), 1,
                                   filter_size, filter_size, filter_size))
            for idx, sigma in enumerate(self.gaussian_sigmas):
                g_masks[idx, 0, :, :, :] = self._fspecial_gauss_3d(filter_size, sigma)
        else:
            # 2D 高斯核
            g_masks = torch.zeros((len(self.gaussian_sigmas), 1,
                                   filter_size, filter_size))
            for idx, sigma in enumerate(self.gaussian_sigmas):
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)

        self.g_masks = g_masks

    def _fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).float()
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def _fspecial_gauss_2d(self, size, sigma):
        g1 = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(g1, g1)

    def _fspecial_gauss_3d(self, size, sigma):
        g1 = self._fspecial_gauss_1d(size, sigma)
        g3d = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
        g3d /= g3d.sum()
        return g3d

    def mean_filter(self, x, window_size=3):
        if x.ndim == 4:  # 2D
            kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
            return F.conv2d(x, kernel.to(x.device), padding=window_size // 2)
        else:  # 3D
            kernel = torch.ones(1, 1, window_size, window_size, window_size) / (window_size ** 3)
            return F.conv3d(x, kernel.to(x.device), padding=window_size // 2)

    def forward(self, obs, syn):
        """
        obs, syn:
            2D: [B, T, R]
            3D: [B, Z, T, R]
        """
        # 归一化
        syn = syn / (1e-10 + torch.max(torch.abs(syn), dim=1, keepdim=True).values)
        obs = obs / (1e-10 + torch.max(torch.abs(obs), dim=1, keepdim=True).values)

        # 判断维度
        if obs.ndim == 3:  # (B,T,R)
            obs = obs.unsqueeze(1)  # (B,1,T,R)
            syn = syn.unsqueeze(1)
            conv = F.conv2d
            pad = self.pad
            if self.is_3d:  # 如果初始化是3D但输入是2D，重新建核
                self._build_kernels(False)
        elif obs.ndim == 4:  # (B,Z,T,R)
            obs = obs.unsqueeze(1)  # (B,1,Z,T,R)
            syn = syn.unsqueeze(1)
            conv = F.conv3d
            pad = (self.pad,) * 3
            if not self.is_3d:  # 如果初始化是2D但输入是3D，重新建核
                self._build_kernels(True)
        else:
            raise ValueError("Input must be (B,T,R) or (B,Z,T,R)")

        self.g_masks = self.g_masks.to(obs.device)

        # 高斯卷积
        mux = conv(obs, self.g_masks, padding=pad, groups=1)
        muy = conv(syn, self.g_masks, padding=pad, groups=1)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = conv(obs * obs, self.g_masks, padding=pad, groups=1) - mux2
        sigmay2 = conv(syn * syn, self.g_masks, padding=pad, groups=1) - muy2
        sigmaxy = conv(obs * syn, self.g_masks, padding=pad, groups=1) - muxy

        L  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)
        CS = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        loss_ms_ssim = torch.mean(-(L * CS))
        loss_mix = self.alpha * loss_ms_ssim * self.compensation

        return loss_mix.mean() + 200
