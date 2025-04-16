import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from lib.common.renderer import Renderer
from lib.provider import near_head_poses
from lib.common.utils import load_config
from lib.common.obj import Mesh

# 创建输出目录
output_dir = "wrist_view_test"
os.makedirs(output_dir, exist_ok=True)

# 加载配置文件
cfg = load_config('configs/default.yaml')

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 渲染参数
H, W = 512, 512
near, far = 0.01, 1000

# 初始化渲染器
renderer = Renderer()

# 直接加载OBJ文件
mesh_path = "mesh.obj"  # 确保这个路径是正确的
print(f"加载人体模型: {mesh_path}")

# 检查文件是否存在
if not os.path.exists(mesh_path):
    print(f"警告: {mesh_path} 不存在!")
    # 尝试使用其他可能的OBJ文件路径
    alternative_paths = [
        "data/mesh.obj",
        "../data/mesh.obj",
        "data/init_body/mesh.obj",
        "../data/init_body/mesh.obj"
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            mesh_path = alt_path
            print(f"找到替代文件: {mesh_path}")
            break
    else:
        print("未找到任何OBJ文件。请确保mesh.obj文件存在于正确的位置。")
        exit(1)

# 加载OBJ文件
mesh = Mesh.load_obj(mesh_path, init_empty_tex=True)
print(f"成功加载网格: {len(mesh.v)}个顶点")

# --- 参数配置 ---
# 基础参数 (左右手通用)
wrist_scale = 1.5  # 缩小视角范围，放大手腕区域
theta_range = [75, 95]  # 更水平的视角
radius_range = [0.2, 0.25]  # 更近距离观察手腕
fov = 50  # 视场角

# 左手特定参数 (调整这些以优化左手视角)
left_wrist_center_coords = [-0.25, 0.1, 0.05]  # 左手腕中心坐标 [x, y, z]
left_view_angles = {
    'front': 0,      # 正面 phi
    'side': -90,     # 左手侧面 phi (通常为负)
    'back': 180,     # 背面 phi
}

# --- 循环处理左右手 ---
for wrist_side in ['left', 'right']:

    # 根据左右手计算当前参数
    if wrist_side == 'left':
        wrist_center_coords = left_wrist_center_coords
        view_angles = left_view_angles
        wrist_center = torch.tensor(wrist_center_coords, device=device).view(1, 3)
    else: # wrist_side == 'right'
        # 对称生成右手参数
        wrist_center_coords = [-left_wrist_center_coords[0], left_wrist_center_coords[1], left_wrist_center_coords[2]]
        view_angles = {
            'front': -left_view_angles['front'], # 0 的相反数还是 0
            'side': -left_view_angles['side'],   # 侧视角取反
            'back': left_view_angles['back'] if left_view_angles['back'] == 180 else -left_view_angles['back'] # 背面通常保持180或-180，也可取反
        }
        wrist_center = torch.tensor(wrist_center_coords, device=device).view(1, 3)

    print(f"\n--- 开始处理 {wrist_side} 手腕 ---")
    print(f"中心点: {wrist_center_coords}")
    print(f"视角 Phi: {view_angles}")

    # 创建一个可视化的结果图
    result_img = np.ones((H*len(view_angles), W*2, 3), dtype=np.uint8) * 255

    # 对每个视角生成渲染图像
    for i, (view_name, phi) in enumerate(view_angles.items()):
        # 使用固定的phi角度，并基于此创建小范围
        phi_range = [phi-5, phi+5]
        
        # 生成相机位置
        poses, dirs, thetas, phis, radius = near_head_poses(
            1,
            device,
            return_dirs=True,
            radius_range=radius_range,
            phi_range=phi_range,
            theta_range=theta_range,
            angle_overhead=30, # 这些可以保持不变或也设为变量
            angle_front=60,  # 这些可以保持不变或也设为变量
            jitter=False,
            shift=wrist_center,
            face_scale=wrist_scale # 注意: near_head_poses 可能内部处理 face_scale，检查其逻辑
        )
        
        # 计算视图矩阵
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        
        projection = torch.tensor([
            [2 * focal / W, 0, 0, 0],
            [0, -2 * focal / H, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=device)
        
        mvp = projection @ torch.inverse(poses)
        
        # 渲染模型
        rgb, normals, alpha = renderer(mesh, mvp, H, W, None, 1, "albedo")
        
        # 转换为可显示格式
        rgb_vis = (rgb[0].cpu().numpy() * 255).astype(np.uint8)
        normals_vis = ((normals[0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        
        # 合并RGB和法线图像
        combined = np.hstack([rgb_vis, normals_vis])
        
        # 添加视角信息
        cv2.putText(combined, f"{wrist_side} {view_name} view (phi={phi}°)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"theta={thetas.item():.1f}°, radius={radius.item():.2f}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存单独视角图像
        output_path = os.path.join(output_dir, f"{wrist_side}_wrist_{view_name}_view.png")
        cv2.imwrite(output_path, combined[:, :, ::-1])  # BGR转RGB
        
        # 添加到结果图像
        result_img[i*H:(i+1)*H, :, :] = combined
        
        print(f"已渲染 {wrist_side} {view_name} 视角:")
        print(f"  - phi: {phi}°")
        print(f"  - theta: {thetas.item():.1f}°")
        print(f"  - radius: {radius.item():.2f}")

    # 保存组合图像
    combined_path = os.path.join(output_dir, f"{wrist_side}_wrist_all_views.png")
    cv2.imwrite(combined_path, result_img[:, :, ::-1])  # BGR转RGB

print(f"完成所有视角的渲染。图像保存在 {output_dir} 目录")
print("\n推荐视角参数配置:")
print("根据渲染图像选择最佳参数，然后在provider.py中更新:")
print("""
elif self.train_wrist:
    camera_type = "wrist"
    
    # 根据当前处理的是左手还是右手来选择适当的中心点
    if random.random() < 0.5:  # 50%概率选择右手
        wrist_shift = torch.as_tensor([0.25, -0.05, 0], device=self.device).view(1, 3)
        phi_range = [85, 95]  # 右手侧视角
    else:  # 50%概率选择左手
        wrist_shift = torch.as_tensor([-0.25, -0.05, 0], device=self.device).view(1, 3)
        phi_range = [-95, -85]  # 左手侧视角
    
    poses, dirs, thetas, phis, radius = near_head_poses(
        1,
        self.device,
        return_dirs=self.opt.dir_text,
        phi_range=phi_range,  # 根据左右手设置合适的水平视角范围
        theta_range=[75, 95],  # 更水平的视角以便看到手腕
        angle_overhead=self.opt.angle_overhead,
        angle_front=self.opt.angle_front,
        jitter=self.opt.jitter_pose,
        shift=wrist_shift,  # 根据左右手选择的中心点
        face_scale=self.wrist_scale * 0.5  # 减小这个值可以放大手腕区域
    )
""")