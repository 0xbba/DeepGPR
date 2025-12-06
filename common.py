import torch
import torch.nn.functional as F
from torch import Tensor
import math
from scipy.constants import c
from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0
import math
from typing import Optional

def ricker(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return a Ricker wavelet with the specified central frequency.

    Args:
        freq: The central frequency.
        length: The number of time samples.
        dt: The time sample spacing.
        peak_time: The time (in secs) of the peak amplitude.
        dtype: The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).

    Returns:
        A PyTorch tensor representing the Ricker wavelet.

    """
    if dt == 0:
        raise ValueError("dt cannot be zero.")

    t: torch.Tensor = torch.arange(float(length), dtype=dtype) * dt - peak_time
    y: torch.Tensor = (1 - 2 * math.pi**2 * freq**2 * t**2) * torch.exp(
        -(math.pi**2) * freq**2 * t**2,
    )
    if dtype is not None:
        return y.to(dtype)
    return y

def initialization(device, er,se,mr,source_apmlitudes,source_location,receiver_location,dx,dt,pmlthick):
    dtype=torch.float32

    if len(er.shape) != 3 and len(er.shape) == 2:
        er.unsqueeze_(-1)
    if len(se.shape) != 3 and len(se.shape) == 2:
        se.unsqueeze_(-1)

    if er.shape == se.shape:
        nx=er.shape[0]
        ny=er.shape[1]
        nz=er.shape[2]
        if nz==1:
            mode=2
        else:
            mode=3
        er=er.to(device)
        se=se.to(device)
        if mr is None:
            mr=torch.ones_like(er, device=device)
        else:
            if mr.shape == er.shape:
                mr=mr.to(device)
            else:
                raise ValueError('The shape of miu should be the same as epsilon and sigma.')
    else:
        raise ValueError('The shape of epsilon and sigma should be the same.')

    if source_location.shape[0] == receiver_location.shape[0]:
        source_location=source_location.to(torch.int)
        receiver_location=receiver_location.to(torch.int)

        source_check = (source_location >= 0).all()
        receiver_check = (receiver_location >= 0).all()

        source_check &= (source_location[..., 0] < nx).all()
        source_check &= (source_location[..., 1] < ny).all()
        source_check &= (source_location[..., 2] < nz).all()

        if not (source_check):
            raise ValueError(
                "Error: Source coordinates out of range! "
                f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
            )
        
        receiver_check &= (receiver_location[..., 0] < nx).all()
        receiver_check &= (receiver_location[..., 1] < ny).all()
        receiver_check &= (receiver_location[..., 2] < nz).all()

        if not (receiver_check):
            raise ValueError(
                "Error: Receiver coordinates out of range! "
                f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
            )
        nstep=source_location.shape[0]

        nsr=source_location.shape[1]
        nrx=receiver_location.shape[1]
        source_apmlitudes=source_apmlitudes.to(device).contiguous()
        source_location=source_location.to(device)
        receiver_location=receiver_location.to(device)
    else:
        raise ValueError('The first dimension (nstep) of source_location and receiver_location should be the same.')
    
    if (source_apmlitudes.shape[0]>1 and source_apmlitudes.shape[0]<nsr) or source_apmlitudes.shape[0]>nsr :
        raise ValueError('The number of source waveforms is incorrect.')
    
    elif source_apmlitudes.shape[0]==1 and nsr!=1:
        source_apmlitudes=source_apmlitudes.repeat(nsr,1,1).contiguous()

    check_cfl(dx, dt,nx,ny,nz)
    nt=source_apmlitudes.shape[1]
    

    pmlthick=pmlthick_revert(pmlthick,er)
    ere=F.pad(er, (0, 1, 0, 1, 0, 1)).to(dtype)
    see=F.pad(se, (0, 1, 0, 1, 0, 1)).to(dtype)
    mr=F.pad(mr, (0, 1, 0, 1, 0, 1)).to(dtype)

    return nx,ny,nz,nt,nstep,nsr,nrx,ere,see,mr,mode,dtype,pmlthick,source_apmlitudes


def check_cfl(dx, dt, nx,ny,nz):

    dy=dx
    dz=dx

    if nz==1:
        dt_max = 1.0 / (c * math.sqrt(1/dx**2 + 1/dy**2))
    elif nx==1:
        dt_max = 1.0 / (c * math.sqrt(1/dy**2 + 1/dz**2))
    elif ny==1:
        dt_max = 1.0 / (c * math.sqrt(1/dx**2 + 1/dz**2))
    else:
        dt_max = 1.0 / (c * math.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))

    if dt > dt_max:
        raise ValueError(f"Does not meet CFL conditions: dt={dt:.3e} > dt_max={dt_max:.3e}")


def pmlthick_revert(p, er):
    if isinstance(p, int):  # 如果是 int
        if er.shape[2] == 1:
            return torch.tensor([p, p, p, p, 0, 0], dtype=torch.int32)
        return torch.tensor([p]*6, dtype=torch.int32)
    
    elif isinstance(p, list):
        if len(p) == 6:
            return torch.tensor(p, dtype=torch.int32)
        elif len(p) == 4:
            return torch.tensor(p + [0, 0], dtype=torch.int32)
        else:
            raise ValueError(f"Unsupported list length: {len(p)}. Must be 4 or 6.")
    
    elif isinstance(p, torch.Tensor):
        return p.to(dtype=torch.int32)
    
    else:
        raise TypeError(f"Unsupported type: {type(p)}")



def tvnorm(tensor,grad,tv):
    if tensor.ndim==3 and tensor.shape[2]==1:
        utensor = total_variation2d(tensor.clone())
        guer=2*(tensor-utensor)
        grad += tv*guer
    else:
        utensor = total_variation_3d(tensor.clone())
        guer=2*(tensor-utensor)
        grad += tv*guer
        
    return grad



def total_variation2d(data, lamda=0.02, rho=1.0, num_iter=500, tol=1e-5):
    # 数据归一化

    data.squeeze_(-1)
    data_max = data.max()
    normalized_data = data / (data_max + 1e-6)
    M, N = normalized_data.shape

    # 初始化变量（带边界）
    X = torch.zeros((M + 2, N + 2), device=data.device)
    X[1:-1, 1:-1] = normalized_data
    Y = X.clone()

    # 辅助变量和乘子
    Zx = torch.zeros_like(X)
    Zy = torch.zeros_like(X)
    Ux = torch.zeros_like(X)
    Uy = torch.zeros_like(X)

    # 创建四邻域卷积核
    kernel = torch.tensor([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=torch.float32, device=data.device).view(1, 1, 3, 3)

    for _ in range(num_iter):
        X_prev = X.clone()

        # 计算差分项
        # 水平方向差分 (循环边界)
        Dxt_Zx = torch.zeros_like(Zx)
        Dxt_Zx[:, :-1] = Zx[:, :-1] - Zx[:, 1:]
        Dxt_Zx[:, -1] = Zx[:, -1] - Zx[:, 0]

        # 垂直方向差分 (循环边界)
        Dyt_Zy = torch.zeros_like(Zy)
        Dyt_Zy[:-1, :] = Zy[:-1, :] - Zy[1:, :]
        Dyt_Zy[-1, :] = Zy[-1, :] - Zy[0, :]

        # 乘子项的差分
        Dxt_Ux = torch.zeros_like(Ux)
        Dxt_Ux[:, :-1] = Ux[:, :-1] - Ux[:, 1:]
        Dxt_Ux[:, -1] = Ux[:, -1] - Ux[:, 0]

        Dyt_Uy = torch.zeros_like(Uy)
        Dyt_Uy[:-1, :] = Uy[:-1, :] - Uy[1:, :]
        Dyt_Uy[-1, :] = Uy[-1, :] - Uy[0, :]

        # 构建RHS
        RHS = Y + lamda * rho * (Dxt_Zx + Dyt_Zy) - lamda * (Dxt_Ux + Dyt_Uy)

        # 使用卷积进行邻域平均
        neighbor_sum = F.conv2d(X[None, None, :, :], kernel, padding=0).squeeze()
        X_center = (neighbor_sum * lamda * rho + RHS[1:-1, 1:-1]) / (1 + 4 * lamda * rho)

        # 更新X的中间区域
        X[1:-1, 1:-1] = X_center

        # 计算梯度项 (带循环边界)
        Dx_X = torch.zeros_like(X)
        Dx_X[:, 1:] = X[:, 1:] - X[:, :-1]
        Dx_X[:, 0] = X[:, 0] - X[:, -1]

        Dy_X = torch.zeros_like(X)
        Dy_X[1:, :] = X[1:, :] - X[:-1, :]
        Dy_X[0, :] = X[0, :] - X[-1, :]

        # 更新Z变量 (软阈值)
        Tx = (Ux + rho * Dx_X) / rho
        Zx = torch.sign(Tx) * torch.clamp(torch.abs(Tx) - 1 / rho, min=0)

        Ty = (Uy + rho * Dy_X) / rho
        Zy = torch.sign(Ty) * torch.clamp(torch.abs(Ty) - 1 / rho, min=0)

        # 更新乘子
        Ux += rho * (Dx_X - Zx)
        Uy += rho * (Dy_X - Zy)

        # 收敛判断
        if torch.norm(X - X_prev) < tol:
            break

    # 返回去噪结果并恢复原始范围
    return (X[1:-1, 1:-1] * data_max).unsqueeze(-1)

def total_variation_3d(
    data: torch.Tensor,
    lamda: float = 0.02,
    rho: float = 1.0,
    num_iter: int = 500,
    tol: float = 1e-5
) -> torch.Tensor:
    """
    3D TV 去噪（ADMM，各向异性分方向 shrink；周期边界）
    仅支持输入 [D, H, W]，输出保持 [D, H, W]
    """
    assert data.ndim == 3, "仅支持三维输入 [D, H, W]"
    dev, dtype = data.device, data.dtype
    D, H, W = data.shape

    # 归一化
    data_max = torch.max(data)
    norm = data / (data_max + 1e-6)

    # 变量初始化（与数据同形状）
    X = norm.clone()
    Y = X.clone()
    Zx = torch.zeros_like(X)
    Zy = torch.zeros_like(X)
    Zz = torch.zeros_like(X)
    Ux = torch.zeros_like(X)
    Uy = torch.zeros_like(X)
    Uz = torch.zeros_like(X)

    def forward_diffs_periodic(X):
        """前向差分（周期边界）"""
        Dx = X - torch.roll(X, shifts=1, dims=2)  # x: W 方向（最后一维）
        Dy = X - torch.roll(X, shifts=1, dims=1)  # y: H 方向
        Dz = X - torch.roll(X, shifts=1, dims=0)  # z: D 方向
        return Dx, Dy, Dz

    def divergence_periodic(Tx, Ty, Tz):
        """
        散度（Dᵗ），对应 2D 版里的 Dxt_*, Dyt_*：
        DᵗT = (T - roll(T, -1)) 在各维度相加
        等价写法，也常见的是 (roll(T, +1) - T)，只要与 forward 保持伴随关系即可
        """
        dxt = Tx - torch.roll(Tx, shifts=-1, dims=2)
        dyt = Ty - torch.roll(Ty, shifts=-1, dims=1)
        dzt = Tz - torch.roll(Tz, shifts=-1, dims=0)
        return dxt + dyt + dzt

    def six_neighbor_sum_periodic(X):
        """6 邻域周期求和：roll ±1 沿三个维度"""
        xm = torch.roll(X, shifts=-1, dims=2)
        xp = torch.roll(X, shifts=+1, dims=2)
        ym = torch.roll(X, shifts=-1, dims=1)
        yp = torch.roll(X, shifts=+1, dims=1)
        zm = torch.roll(X, shifts=-1, dims=0)
        zp = torch.roll(X, shifts=+1, dims=0)
        return xm + xp + ym + yp + zm + zp

    for _ in range(num_iter):
        X_prev = X

        # RHS = Y + λρ Dᵗ Z - λ Dᵗ U
        divZ = divergence_periodic(Zx, Zy, Zz)
        divU = divergence_periodic(Ux, Uy, Uz)
        RHS = Y + lamda * rho * divZ - lamda * divU

        # X 更新： (1 + 6λρ) X = λρ * neighbor_sum(X) + RHS
        neighbor_sum = six_neighbor_sum_periodic(X)
        X = (lamda * rho * neighbor_sum + RHS) / (1.0 + 6.0 * lamda * rho)

        # 前向差分（用于 Z/U）
        Dx, Dy, Dz = forward_diffs_periodic(X)

        # Z：软阈值（分方向各向异性，与你的 2D 写法一致）
        Tx = (Ux + rho * Dx) / rho
        Ty = (Uy + rho * Dy) / rho
        Tz = (Uz + rho * Dz) / rho
        Zx = torch.sign(Tx) * torch.clamp(torch.abs(Tx) - 1.0 / rho, min=0.0)
        Zy = torch.sign(Ty) * torch.clamp(torch.abs(Ty) - 1.0 / rho, min=0.0)
        Zz = torch.sign(Tz) * torch.clamp(torch.abs(Tz) - 1.0 / rho, min=0.0)

        # U：U += ρ (D X - Z)
        Ux = Ux + rho * (Dx - Zx)
        Uy = Uy + rho * (Dy - Zy)
        Uz = Uz + rho * (Dz - Zz)

        # 收敛判断
        if torch.norm(X - X_prev) < tol:
            break

    # 还原幅值并返回 3D
    return X * data_max




def create_or_separate(tensor:tuple, nx,ny,nz,nstep,device: torch.device,
                  dtype: torch.dtype):
    if tensor == None:
        return torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous(),torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous(),torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous()
    # else:
    #     if tensor[0].shape[1]==nx+1 and tensor[0].shape[2]==ny+1 and tensor[0].shape[3]==nz+1 and tensor[0].shape[0]==nstep:
    #       return tensor[0].contiguous(),tensor[1].contiguous(),tensor[2].contiguous()
    #     else:
    #       print(nstep,nx,ny,nz)
    #       raise ValueError('The shape of E and H should be (nstep,nx+1,ny+1,nz+1).')
    else:
        if (
            tensor[0].shape[1] == nx + 1
            and tensor[0].shape[2] == ny + 1
            and tensor[0].shape[3] == nz + 1
            and tensor[0].shape[0] == nstep
        ):
            return (
                tensor[0].contiguous(),
                tensor[1].contiguous(),
                tensor[2].contiguous()
            )
        else:
            expected = (nstep, nx + 1, ny + 1, nz + 1)
            actual = (tuple(tensor[0].shape), tuple(tensor[1].shape), tuple(tensor[2].shape))
            raise ValueError(f"Expected {expected}, but got {actual}.")



def check_tensors_for_nan_inf(**tensors):
    """
        check_tensors_for_nan_inf(
            gEx=gEx, gEy=gEy, gEz=gEz,
            gHx=gHx, gHy=gHy, gHz=gHz,
            gx0EPhi1=gx0EPhi1, ...
        )
    """
    found_issue = False

    for name, tensor in tensors.items():
        if tensor is None:
            print(f"[WARNING] {name} is None.")
            continue

        if not isinstance(tensor, torch.Tensor):
            print(f"[WARNING] {name} is not a tensor: {type(tensor)}")
            continue

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            found_issue = True
            print(f"❌ [ERROR] Tensor `{name}` contains:", end=" ")
            if has_nan:
                print("NaN ", end="")
            if has_inf:
                print("Inf ", end="")
            print(f"| shape={tuple(tensor.shape)} | dtype={tensor.dtype}")











def buildpmlcoeffs(er,mr,dt,dx,nx,ny,nz,pmlthick,device,dtype):
    averageer=torch.zeros(6, device=device, dtype=dtype)
    averagemr=torch.zeros(6, device=device, dtype=dtype)
    lencfs=1
    x0 = torch.empty(0)
    xm = torch.empty(0)
    y0 = torch.empty(0)
    ym = torch.empty(0)
    z0 = torch.empty(0)
    zm = torch.empty(0)
    x01 = torch.empty(0)
    x02 = torch.empty(0)
    xm1 = torch.empty(0)
    xm2 = torch.empty(0)
    y01 = torch.empty(0)
    y02 = torch.empty(0)
    ym1 = torch.empty(0)
    ym2 = torch.empty(0)
    z01 = torch.empty(0)
    z02 = torch.empty(0)
    zm1 = torch.empty(0)
    zm2 = torch.empty(0)
    if pmlthick[0]>0:
        x0=torch.tensor((pmlthick[0],0,pmlthick[0],0,ny,0,nz), device=device, dtype=torch.int)
        averageer[0]=er[x0[1],:ny,:nz].mean()
        averagemr[0]=mr[x0[1],:ny,:nz].mean()
        CFS0=CFS()
        x01=torch.zeros((4,lencfs,pmlthick[0]), device=device, dtype=dtype)
        x02=torch.zeros((4,lencfs,pmlthick[0]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS0,x01,x02, averageer[0], averagemr[0], dt,dx,pmlthick[0])

    if pmlthick[1]>0:
        xm=torch.tensor((pmlthick[1],nx-pmlthick[1],nx,0,ny,0,nz), device=device, dtype=torch.int)
        averageer[1]=er[xm[1],:ny,:nz].mean()
        averagemr[1]=mr[xm[1],:ny,:nz].mean()
        CFS1=CFS()
        xm1=torch.zeros((4,lencfs,pmlthick[1]), device=device, dtype=dtype)
        xm2=torch.zeros((4,lencfs,pmlthick[1]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS1,xm1,xm2, averageer[1], averagemr[1], dt,dx,pmlthick[1])

    if pmlthick[2]>0:
        y0=torch.tensor((pmlthick[2],0,nx,0,pmlthick[2],0,nz), device=device, dtype=torch.int)
        averageer[2]=er[:nx,y0[3],:nz].mean()
        averagemr[2]=mr[:nx,y0[3],:nz].mean()
        CFS2=CFS()
        y01=torch.zeros((4,lencfs,pmlthick[2]), device=device, dtype=dtype)
        y02=torch.zeros((4,lencfs,pmlthick[2]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS2,y01,y02, averageer[2], averagemr[2], dt,dx,pmlthick[2])

    if pmlthick[3]>0:
        ym=torch.tensor((pmlthick[3],0,nx,ny-pmlthick[3],ny,0,nz), device=device, dtype=torch.int)
        averageer[3]=er[:nx,ym[3],:nz].mean()
        averagemr[3]=mr[:nx,ym[3],:nz].mean()
        CFS3=CFS()
        ym1=torch.zeros((4,lencfs,pmlthick[3]), device=device, dtype=dtype)
        ym2=torch.zeros((4,lencfs,pmlthick[3]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS3,ym1,ym2, averageer[3], averagemr[3], dt,dx,pmlthick[3])

    if pmlthick[4]>0:
        z0=torch.tensor((pmlthick[4],0,nx,0,ny,0,pmlthick[4]), device=device, dtype=torch.int)
        averageer[4]=er[:nx,:ny,z0[5]].mean()
        averagemr[4]=mr[:nx,:ny,z0[5]].mean()
        CFS4=CFS()
        z01=torch.zeros((4,lencfs,pmlthick[4]), device=device, dtype=dtype)
        z02=torch.zeros((4,lencfs,pmlthick[4]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS4,z01,z02, averageer[4], averagemr[4], dt,dx,pmlthick[4])

    if pmlthick[5]>0:
        zm=torch.tensor((pmlthick[5],0,nx,0,ny,nz-pmlthick[5],nz), device=device, dtype=torch.int)
        averageer[5]=er[:nx,:ny,zm[5]].mean()
        averagemr[5]=mr[:nx,:ny,zm[5]].mean()
        CFS5=CFS()
        zm1=torch.zeros((4,lencfs,pmlthick[5]), device=device, dtype=dtype)
        zm2=torch.zeros((4,lencfs,pmlthick[5]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS5,zm1,zm2, averageer[5], averagemr[5], dt,dx,pmlthick[5])
    return x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2




class CFSParameter(object):
    scalingprofiles = {'constant': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4, 'quintic': 5, 'sextic': 6, 'septic': 7, 'octic': 8}

    def __init__(self,ID =None, scaling='polynomial', scalingprofile=None, min=0, max=0):
        self.ID = ID
        self.scaling = scaling
        self.scalingprofile = scalingprofile
        self.min = min
        self.max = max


class CFS(object):

    def __init__(self):
        self.alpha = CFSParameter(ID='alpha', scalingprofile='constant')
        self.kappa = CFSParameter(ID='kappa', scalingprofile='constant', min=1, max=1)
        self.sigma = CFSParameter(ID='sigma', scalingprofile='quartic', min=0, max=None)
        self.device = torch.device("cuda")

    def calculate_sigmamax(self, d, er, mr):
        with torch.no_grad():
            m = CFSParameter.scalingprofiles[self.sigma.scalingprofile]
            self.sigma.max = (0.8 * (m + 1)) / (((m0 / e0) ** 0.5) * d * torch.sqrt(er * mr))




    def scaling_polynomial(self, order, Evalues, Hvalues):
        tmp = (torch.linspace(0, (len(Evalues) - 1) + 0.5, steps=2 * len(Evalues)) / (len(Evalues) - 1)) ** order
        Evalues = tmp[0:-1:2].to(self.device)
        Hvalues = tmp[1::2].to(self.device)
        return Evalues, Hvalues

    def calculate_values(self, thickness, parameter):

        Evalues = torch.zeros(thickness + 1, device=self.device)
        Hvalues = torch.zeros(thickness + 1, device=self.device)
        if parameter.scalingprofile == 'constant':
            Evalues += parameter.max
            Hvalues += parameter.max
        elif parameter.scaling == 'polynomial':
            Evalues, Hvalues = self.scaling_polynomial(CFSParameter.scalingprofiles[parameter.scalingprofile], Evalues, Hvalues)
            if parameter.ID == 'alpha':
                Evalues = Evalues * (self.alpha.max - self.alpha.min) + self.alpha.min
                Hvalues = Hvalues * (self.alpha.max - self.alpha.min) + self.alpha.min
            elif parameter.ID == 'kappa':
                Evalues = Evalues * (self.kappa.max - self.kappa.min) + self.kappa.min
                Hvalues = Hvalues * (self.kappa.max - self.kappa.min) + self.kappa.min
            elif parameter.ID == 'sigma':
                Evalues = Evalues * (self.sigma.max - self.sigma.min) + self.sigma.min
                Hvalues = Hvalues * (self.sigma.max - self.sigma.min) + self.sigma.min

        Evalues = Evalues[:-1]
        Hvalues = Hvalues[:-1]
        
        return Evalues, Hvalues

def calculate_pml_update_coeffs(cfs,R1,R2, aver, avmr, dt,d,thickness):
    if not cfs.sigma.max:
        cfs.calculate_sigmamax(d, aver, avmr)

    Ealpha, Halpha = cfs.calculate_values(thickness, cfs.alpha)
    Ekappa, Hkappa = cfs.calculate_values(thickness, cfs.kappa)
    Esigma, Hsigma = cfs.calculate_values(thickness, cfs.sigma)

    R1=R1.contiguous()
    R2=R2.contiguous()

    tmp = (2 * e0 * Ekappa) + dt * (Ealpha * Ekappa + Esigma)
    R1[0,0, :] = (2 * e0 + dt * Ealpha) / tmp
    R1[1,0, :] = (2 * e0 * Ekappa) / tmp
    R1[2,0, :] = ((2 * e0 * Ekappa) - dt * (Ealpha * Ekappa + Esigma)) / tmp
    R1[3,0, :] = (2 * Esigma * dt) / (Ekappa * tmp)
    # print(R1)

    tmp = (2 * e0 * Hkappa) + dt * (Halpha * Hkappa + Hsigma)
    R2[0,0, :] = (2 * e0 + dt * Halpha) / tmp
    R2[1,0, :] = (2 * e0 * Hkappa) / tmp
    R2[2,0, :] = ((2 * e0 * Hkappa) - dt * (Halpha * Hkappa + Hsigma)) / tmp
    R2[3,0, :] = (2 * Hsigma * dt) / (Hkappa * tmp)
    # print(R2)




def build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device):
   
    (x0EPhi1, x0EPhi2, x0HPhi1, x0HPhi2,
    xmEPhi1, xmEPhi2, xmHPhi1, xmHPhi2,
    y0EPhi1, y0EPhi2, y0HPhi1, y0HPhi2,
    ymEPhi1, ymEPhi2, ymHPhi1, ymHPhi2,
    z0EPhi1, z0EPhi2, z0HPhi1, z0HPhi2,
    zmEPhi1, zmEPhi2, zmHPhi1, zmHPhi2) = [torch.empty(0) for _ in range(24)]

    if PML==None:
        PML=(None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None)

    if x0.numel()!=0 and PML[0]==None:
        x0EPhi1=torch.zeros((nstep, int(x0[2]-x0[1]+1), int(x0[4]-x0[3]), int(x0[6]-x0[5]+1)),dtype=torch.float, device=device)
        x0EPhi2=torch.zeros((nstep, int(x0[2]-x0[1]+1), int(x0[4]-x0[3]+1), int(x0[6]-x0[5])),dtype=torch.float, device=device)
        x0HPhi1=torch.zeros((nstep, int(x0[2]-x0[1]), int(x0[4]-x0[3]+1), int(x0[6]-x0[5])), dtype=torch.float, device=device)
        x0HPhi2=torch.zeros((nstep, int(x0[2]-x0[1]), int(x0[4]-x0[3]), int(x0[6]-x0[5]+1)), dtype=torch.float, device=device)
    elif PML[0]!=None:
        x0EPhi1=PML[0].contiguous()
        x0EPhi2=PML[1].contiguous()
        x0HPhi1=PML[2].contiguous()
        x0HPhi2=PML[3].contiguous()

    if xm.numel()!=0 and PML[4]==None:
        xmEPhi1=torch.zeros((nstep, xm[2]-xm[1]+1, xm[4]-xm[3], xm[6]-xm[5]+1), dtype=torch.float, device=device)
        xmEPhi2=torch.zeros((nstep, xm[2]-xm[1]+1, xm[4]-xm[3]+1, xm[6]-xm[5]), dtype=torch.float, device=device)
        xmHPhi1=torch.zeros((nstep, xm[2]-xm[1], xm[4]-xm[3]+1, xm[6]-xm[5]), dtype=torch.float, device=device)
        xmHPhi2=torch.zeros((nstep, xm[2]-xm[1], xm[4]-xm[3], xm[6]-xm[5]+1), dtype=torch.float, device=device)    
    elif PML[4]!=None:
        xmEPhi1=PML[4].contiguous()
        xmEPhi2=PML[5].contiguous()
        xmHPhi1=PML[6].contiguous()
        xmHPhi2=PML[7].contiguous()

    if y0.numel()!=0 and PML[8]==None:
        y0EPhi1=torch.zeros((nstep, y0[2]-y0[1], y0[4]-y0[3]+1, y0[6]-y0[5]+1), dtype=torch.float, device=device)
        y0EPhi2=torch.zeros((nstep, y0[2]-y0[1]+1, y0[4]-y0[3]+1, y0[6]-y0[5]), dtype=torch.float, device=device)
        y0HPhi1=torch.zeros((nstep, y0[2]-y0[1]+1, y0[4]-y0[3], y0[6]-y0[5]), dtype=torch.float, device=device)
        y0HPhi2=torch.zeros((nstep, y0[2]-y0[1], y0[4]-y0[3], y0[6]-y0[5]+1), dtype=torch.float, device=device)
    elif PML[8]!=None:
        y0EPhi1=PML[8].contiguous()
        y0EPhi2=PML[9].contiguous()
        y0HPhi1=PML[10].contiguous()
        y0HPhi2=PML[11].contiguous()

    if ym.numel()!=0 and PML[12]==None:
        ymEPhi1=torch.zeros((nstep, ym[2]-ym[1], ym[4]-ym[3]+1, ym[6]-ym[5]+1),dtype=torch.float, device=device)
        ymEPhi2=torch.zeros((nstep, ym[2]-ym[1]+1, ym[4]-ym[3]+1, ym[6]-ym[5]), dtype=torch.float, device=device)
        ymHPhi1=torch.zeros((nstep, ym[2]-ym[1]+1, ym[4]-ym[3], ym[6]-ym[5]), dtype=torch.float, device=device)
        ymHPhi2=torch.zeros((nstep, ym[2]-ym[1], ym[4]-ym[3], ym[6]-ym[5]+1), dtype=torch.float, device=device)
    elif PML[12]!=None:
        ymEPhi1=PML[12].contiguous()
        ymEPhi2=PML[13].contiguous()
        ymHPhi1=PML[14].contiguous()
        ymHPhi2=PML[15].contiguous()

    if z0.numel()!=0 and PML[16]==None:
        z0EPhi1=torch.zeros((nstep, z0[2]-z0[1], z0[4]-z0[3]+1, z0[6]-z0[5]+1), dtype=torch.float, device=device)
        z0EPhi2=torch.zeros((nstep, z0[2]-z0[1]+1, z0[4]-z0[3], z0[6]-z0[5]+1), dtype=torch.float, device=device)
        z0HPhi1=torch.zeros((nstep, z0[2]-z0[1]+1, z0[4]-z0[3], z0[6]-z0[5]), dtype=torch.float, device=device)
        z0HPhi2=torch.zeros((nstep, z0[2]-z0[1], z0[4]-z0[3]+1, z0[6]-z0[5]), dtype=torch.float, device=device)
    elif PML[16]!=None:
        z0EPhi1=PML[16].contiguous()
        z0EPhi2=PML[17].contiguous()
        z0HPhi1=PML[18].contiguous()
        z0HPhi2=PML[19].contiguous()

    if zm.numel()!=0 and PML[20]==None:
        zmEPhi1=torch.zeros((nstep, zm[2]-zm[1], zm[4]-zm[3]+1, zm[6]-zm[5]+1), dtype=torch.float, device=device)
        zmEPhi2=torch.zeros((nstep, zm[2]-zm[1]+1, zm[4]-zm[3], zm[6]-zm[5]+1), dtype=torch.float, device=device)
        zmHPhi1=torch.zeros((nstep, zm[2]-zm[1]+1, zm[4]-zm[3], zm[6]-zm[5]), dtype=torch.float, device=device)
        zmHPhi2=torch.zeros((nstep, zm[2]-zm[1], zm[4]-zm[3]+1, zm[6]-zm[5]), dtype=torch.float, device=device)
    elif PML[20]!=None:
        zmEPhi1=PML[20].contiguous()
        zmEPhi2=PML[21].contiguous()
        zmHPhi1=PML[22].contiguous()
        zmHPhi2=PML[23].contiguous()

    return x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2


def checkpoint_initial_field(device=None,per_nstep=None, dx=None, dt=None, 
            source_apmlitudes=None,
            source_location=None, 
            receiver_location=None, 
            er=None, se=None,mr=None, 
            pmlthick=10):
    E=None
    H=None
    PML=None

    nx,ny,nz,nt,nstep,nsr,nrx,ere,see,mr,mode,dtype,pmlthick,source_apmlitudes=initialization(device,er,se,mr,source_apmlitudes,source_location,receiver_location,dx,dt,pmlthick)

    Ex,Ey,Ez=create_or_separate(E,nx,ny,nz,nstep,device,dtype)
    Hx,Hy,Hz=create_or_separate(H,nx,ny,nz,nstep,device,dtype)

    x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2=buildpmlcoeffs(er,mr,dt,dx,nx,ny,nz,pmlthick,device,dtype)


    x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2=build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device)

    del x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2

    print(per_nstep)
    print(nstep)
    if per_nstep==None:
        return (Ex,Ey,Ez),(Hx,Hy,Hz),(x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2)
    else:
        return (Ex[:per_nstep,:,:,:],Ey[:per_nstep,:,:,:],Ez[:per_nstep,:,:,:]),(Hx[:per_nstep,:,:,:],Hy[:per_nstep,:,:,:],Hz[:per_nstep,:,:,:]),(x0EPhi1[:per_nstep,:,:,:],x0EPhi2[:per_nstep,:,:,:],x0HPhi1[:per_nstep,:,:,:],x0HPhi2[:per_nstep,:,:,:],xmEPhi1[:per_nstep,:,:,:],xmEPhi2[:per_nstep,:,:,:],xmHPhi1[:per_nstep,:,:,:],xmHPhi2[:per_nstep,:,:,:],y0EPhi1[:per_nstep,:,:,:],y0EPhi2[:per_nstep,:,:,:],y0HPhi1[:per_nstep,:,:,:],y0HPhi2[:per_nstep,:,:,:],ymEPhi1[:per_nstep,:,:,:],ymEPhi2[:per_nstep,:,:,:],ymHPhi1[:per_nstep,:,:,:],ymHPhi2[:per_nstep,:,:,:],z0EPhi1[:per_nstep,:,:,:],z0EPhi2[:per_nstep,:,:,:],z0HPhi1[:per_nstep,:,:,:],z0HPhi2[:per_nstep,:,:,:],zmEPhi1[:per_nstep,:,:,:],zmEPhi2[:per_nstep,:,:,:],zmHPhi1[:per_nstep,:,:,:],zmHPhi2[:per_nstep,:,:,:])


def zero_field(*tensors):
    zeroed_copies = []
    for t in tensors:
        if t is not None:
            zeroed_copies.append(torch.zeros_like(t))
    return zeroed_copies
