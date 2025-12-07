import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_survey_geometry(model, sources, receivers, title="Survey Geometry"):

    # 1. 数据转换与展平 (保持不变)
    if isinstance(model, torch.Tensor):
        model = model.detach().cpu().numpy()
    if isinstance(sources, torch.Tensor):
        sources = sources.detach().cpu().numpy()
    if isinstance(receivers, torch.Tensor):
        receivers = receivers.detach().cpu().numpy()

    # 处理多炮维度 [Shots, N, 3] -> [Total, 3]
    if sources.ndim == 3: sources = sources.reshape(-1, 3)
    if receivers.ndim == 3: receivers = receivers.reshape(-1, 3)

    nx, ny, nz = model.shape
    
    # 2. 绘图
    if nz == 1:
        # ==========================================
        # 2D 绘图模式 (平面图, Z=1)
        # ==========================================
        fig, ax = plt.subplots(figsize=(10, 8))
        
        slice_data = model[:, :, 0].T 
        
        im = ax.imshow(slice_data, cmap='jet', origin='lower', 
                       extent=[0, nx, 0, ny], aspect='auto')
        
        plt.colorbar(im, ax=ax, label='Parameter Value')

        # 绘制源 (x坐标在 sources[:,0], y坐标在 sources[:,1])
        ax.scatter(sources[:, 0], sources[:, 1], c='red', marker='*', s=150, 
                   label='Sources', edgecolors='k', zorder=10)
        
        # 绘制接收器
        ax.scatter(receivers[:, 0], receivers[:, 1], c='white', marker='v', s=80, 
                   label='Receivers', edgecolors='k', zorder=9, alpha=0.7)

        ax.set_xlabel('X Dimension (nx)')
        ax.set_ylabel('Y Dimension (ny)')
        ax.set_title(f"{title} (2D Top View)")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)

    else:
        # ==========================================
        # 3D 绘图模式
        # ==========================================
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 散点图: scatter(x, y, z) 本身就是符合直觉的
        ax.scatter(sources[:, 0], sources[:, 1], sources[:, 2], 
                   c='red', marker='*', s=100, label='Sources')
        ax.scatter(receivers[:, 0], receivers[:, 1], receivers[:, 2], 
                   c='blue', marker='v', s=40, label='Receivers', alpha=0.6)

        # 绘制切片
        x_grid = np.arange(nx)
        y_grid = np.arange(ny)
        z_grid = np.arange(nz)
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        
        # 为了保证切片颜色对应正确，contourf 需要传入网格化坐标
        # Z平面 (底面/中间面)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij') 
        # indexing='ij' 确保 X 对应第一维(nx), Y 对应第二维(ny)
        ax.contourf(X, Y, model[:, :, cz], zdir='z', offset=cz, cmap='viridis', alpha=0.5)
        
        # Y平面 (侧面)
        X, Z = np.meshgrid(x_grid, z_grid, indexing='ij')
        ax.contourf(X, model[:, cy, :], Z, zdir='y', offset=cy, cmap='viridis', alpha=0.5)
        
        # X平面 (正面)
        Y, Z = np.meshgrid(y_grid, z_grid, indexing='ij')
        ax.contourf(model[cx, :, :], Y, Z, zdir='x', offset=cx, cmap='viridis', alpha=0.5)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        ax.set_xlabel('X (nx)')
        ax.set_ylabel('Y (ny)')
        ax.set_zlabel('Z (nz)')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()
