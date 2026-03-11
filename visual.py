import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_survey_geometry(model, sources, receivers, dx=0.02, title="Survey Geometry"):

    # 1. 数据转换与展平
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
        # 2D 绘图模式 (垂直朝上，0在顶部)
        # ==========================================
        fig, ax = plt.subplots(figsize=(6, 4)) 
        
        slice_data = model[:, :, 0] # Shape: (nx, ny)
        
        # [修改 1] extent 乘以 dx，转换为真实物理距离
        # extent=[左, 右, 下, 上] 
        # 垂直轴从 nx*dx (底部) 到 0 (顶部)
        extent_real = [0, ny * dx, nx * dx, 0]
        
        im = ax.imshow(slice_data, cmap='jet', origin='upper', 
                       extent=extent_real, aspect='auto')
        
        plt.colorbar(im, ax=ax, label='Relative permittivity')

        # [修改 2] 散点坐标乘以 dx
        # Scatter x (水平) = sources[:, 1] * dx
        # Scatter y (垂直) = sources[:, 0] * dx
        ax.scatter(sources[:, 1] * dx, sources[:, 0] * dx, c='red', marker='*', s=150, 
                   label='Sources', edgecolors='k', zorder=10)
        
        ax.scatter(receivers[:, 1] * dx, receivers[:, 0] * dx, c='white', marker='v', s=80, 
                   label='Receivers', edgecolors='k', zorder=9, alpha=0.7)

        # [修改 3] 标签改为物理单位
        ax.set_xlabel('Distance (m)') 
        ax.set_ylabel('Depth (m)') 
        
        # 保持 Y 轴方向控制 (0 在上)
        if ax.get_ylim()[0] < ax.get_ylim()[1]: 
             ax.invert_yaxis()

        # ax.set_title(f"{title} ")
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)

    else:
        # ==========================================
        # 3D 绘图模式 (同步应用 dx 以保持一致)
        # ==========================================
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 散点坐标乘以 dx
        ax.scatter(sources[:, 0] * dx, sources[:, 1] * dx, sources[:, 2] * dx, 
                   c='red', marker='*', s=100, label='Sources')
        ax.scatter(receivers[:, 0] * dx, receivers[:, 1] * dx, receivers[:, 2] * dx, 
                   c='blue', marker='v', s=40, label='Receivers', alpha=0.6)

        # 网格坐标乘以 dx
        x_grid = np.arange(nx) * dx
        y_grid = np.arange(ny) * dx
        z_grid = np.arange(nz) * dx
        
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        
        # 生成真实坐标的 Meshgrid
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij') 
        # 注意 offset 也要乘以 dx
        ax.contourf(X, Y, model[:, :, cz], zdir='z', offset=cz * dx, cmap='viridis', alpha=0.5)
        
        X, Z = np.meshgrid(x_grid, z_grid, indexing='ij')
        ax.contourf(X, model[:, cy, :], Z, zdir='y', offset=cy * dx, cmap='viridis', alpha=0.5)
        
        Y, Z = np.meshgrid(y_grid, z_grid, indexing='ij')
        ax.contourf(model[cx, :, :], Y, Z, zdir='x', offset=cx * dx, cmap='viridis', alpha=0.5)

        # 设置真实坐标范围
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(0, ny * dx)
        ax.set_zlim(0, nz * dx)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()