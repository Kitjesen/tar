#!/usr/bin/env python3

"""
快速预览地形 - 直接弹出窗口显示地形图像
不依赖Isaac Lab，直接用matplotlib显示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# 导入地形生成器 (从上级目录)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rough_terrain import RoughTerrainGenerator, create_rough_terrain_configs
from stairs_terrain import StairsTerrainGenerator, create_stairs_terrain_configs
from gap_terrain import GapTerrainGenerator, create_gap_terrain_configs


def plot_3d_terrain(height_map: np.ndarray, title: str, ax=None):
    """绘制3D地形"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # 创建网格
    rows, cols = height_map.shape
    x = np.linspace(0, 8, cols)  # 假设8米宽度
    y = np.linspace(0, 8, rows)  # 假设8米长度
    X, Y = np.meshgrid(x, y)
    
    # 绘制3D表面
    surf = ax.plot_surface(X, Y, height_map, 
                          cmap='terrain', 
                          alpha=0.9,
                          linewidth=0.5,
                          antialiased=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (米)', fontsize=12)
    ax.set_ylabel('Y (米)', fontsize=12) 
    ax.set_zlabel('高度 (米)', fontsize=12)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 添加颜色条
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='高度 (m)')
    
    return ax


def plot_2d_terrain(height_map: np.ndarray, title: str, ax=None):
    """绘制2D地形（俯视图）"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用光照效果增强视觉
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(height_map, cmap=plt.cm.terrain, vert_exag=2.0, blend_mode='soft')
    
    im = ax.imshow(rgb, origin='lower', extent=[0, 8, 0, 8])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (米)', fontsize=12)
    ax.set_ylabel('Y (米)', fontsize=12)
    
    # 添加等高线
    contours = ax.contour(height_map, levels=10, colors='black', alpha=0.3, linewidths=0.5,
                         extent=[0, 8, 0, 8])
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    return ax


def show_terrain_gallery():
    """显示地形画廊"""
    print("🎨 生成地形画廊...")
    
    # 生成4种地形
    terrains = []
    
    # 1. 崎岖地形
    print("  📍 生成崎岖地形...")
    rough_configs = create_rough_terrain_configs()
    rough_gen = RoughTerrainGenerator(rough_configs["medium"])
    rough_map = rough_gen.generate(seed=42)
    terrains.append(("崎岖地形 (随机高度变化)", rough_map))
    
    # 2. 上楼梯
    print("  📍 生成上楼梯...")
    stairs_configs = create_stairs_terrain_configs()
    stairs_gen = StairsTerrainGenerator(stairs_configs["medium_up"])
    stairs_map = stairs_gen.generate(seed=42)
    terrains.append(("上楼梯 (台阶训练)", stairs_map))
    
    # 3. 下楼梯  
    print("  📍 生成下楼梯...")
    stairs_down_gen = StairsTerrainGenerator(stairs_configs["medium_down"])
    stairs_down_map = stairs_down_gen.generate(seed=42)
    terrains.append(("下楼梯 (下坡训练)", stairs_down_map))
    
    # 4. 跳跃障碍
    print("  📍 生成跳跃障碍...")
    gap_configs = create_gap_terrain_configs()
    gap_gen = GapTerrainGenerator(gap_configs["medium_height"])
    gap_map = gap_gen.generate(seed=42)
    terrains.append(("跳跃障碍 (间隙跨越)", gap_map))
    
    # 创建2D画廊
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, (title, height_map) in enumerate(terrains):
        plot_2d_terrain(height_map, title, axes[i])
        
        # 添加统计信息
        min_h, max_h = np.min(height_map), np.max(height_map)
        mean_h, std_h = np.mean(height_map), np.std(height_map)
        
        stats_text = f"""统计信息:
高度: {min_h:.3f}~{max_h:.3f}m
平均: {mean_h:.3f}m  
标准差: {std_h:.3f}m"""
        
        axes[i].text(0.02, 0.98, stats_text, 
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
    
    plt.suptitle('🤖 机器狗训练地形展示', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 创建3D画廊
    fig = plt.figure(figsize=(20, 15))
    
    for i, (title, height_map) in enumerate(terrains):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        plot_3d_terrain(height_map, title, ax)
    
    plt.suptitle('🤖 机器狗训练地形 - 3D视图', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ 地形预览完成!")


def show_difficulty_progression():
    """显示难度进展"""
    print("📈 生成难度进展图...")
    
    # 以崎岖地形为例展示难度进展
    rough_configs = create_rough_terrain_configs()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    difficulties = ['easy', 'medium', 'hard']
    for i, difficulty in enumerate(difficulties):
        generator = RoughTerrainGenerator(rough_configs[difficulty])
        height_map = generator.generate(seed=42)
        
        plot_2d_terrain(height_map, f'崎岖地形 - {difficulty}难度', axes[i])
        
        # 添加难度信息
        max_h = np.max(height_map)
        std_h = np.std(height_map)
        
        diff_text = f"""难度: {difficulty}
最大高度: {max_h:.3f}m
复杂度: {std_h:.3f}m"""
        
        axes[i].text(0.02, 0.98, diff_text,
                    transform=axes[i].transAxes,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('📊 地形难度进展示例 (崎岖地形)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ 难度进展展示完成!")


def interactive_terrain_explorer():
    """交互式地形浏览器"""
    print("🎮 启动交互式地形浏览器...")
    
    # 创建所有地形
    all_terrains = {}
    
    # 崎岖地形
    rough_configs = create_rough_terrain_configs()
    for name, config in rough_configs.items():
        gen = RoughTerrainGenerator(config)
        height_map = gen.generate(seed=42)
        all_terrains[f"崎岖地形_{name}"] = height_map
    
    # 楼梯地形
    stairs_configs = create_stairs_terrain_configs()
    for name, config in stairs_configs.items():
        gen = StairsTerrainGenerator(config)
        height_map = gen.generate(seed=42)
        all_terrains[f"楼梯_{name}"] = height_map
    
    # 跳跃障碍
    gap_configs = create_gap_terrain_configs()
    for name, config in gap_configs.items():
        gen = GapTerrainGenerator(config)
        height_map = gen.generate(seed=42)
        all_terrains[f"障碍_{name}"] = height_map
    
    print(f"📋 可用地形 ({len(all_terrains)}个):")
    for i, name in enumerate(all_terrains.keys(), 1):
        print(f"  {i:2d}. {name}")
    
    while True:
        try:
            choice = input(f"\n选择要查看的地形 (1-{len(all_terrains)}) 或输入 'q' 退出: ")
            
            if choice.lower() == 'q':
                print("👋 退出浏览器")
                break
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(all_terrains):
                terrain_names = list(all_terrains.keys())
                selected_name = terrain_names[choice_idx]
                selected_terrain = all_terrains[selected_name]
                
                print(f"🔍 显示: {selected_name}")
                
                # 显示选中的地形
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # 2D视图
                plot_2d_terrain(selected_terrain, f"{selected_name} - 俯视图", ax1)
                
                # 3D视图
                ax2 = fig.add_subplot(122, projection='3d')
                plot_3d_terrain(selected_terrain, f"{selected_name} - 3D视图", ax2)
                
                plt.tight_layout()
                plt.show()
                
            else:
                print("❌ 无效选择，请重试")
                
        except (ValueError, KeyboardInterrupt):
            print("👋 退出浏览器")
            break


def main():
    """主函数"""
    print("🚀" + "=" * 48 + "🚀")
    print("    🤖 机器狗地形系统 - 快速预览器")
    print("🚀" + "=" * 48 + "🚀")
    
    print("\n📋 选择预览模式:")
    print("  1. 地形画廊 (推荐) - 查看所有主要地形")
    print("  2. 难度进展 - 查看训练难度变化")
    print("  3. 交互浏览 - 逐个查看所有地形")
    print("  4. 全部显示 - 运行所有模式")
    
    while True:
        try:
            choice = input("\n请选择模式 (1-4): ")
            
            if choice == '1':
                show_terrain_gallery()
                break
            elif choice == '2':
                show_difficulty_progression()
                break
            elif choice == '3':
                interactive_terrain_explorer()
                break
            elif choice == '4':
                show_terrain_gallery()
                show_difficulty_progression() 
                interactive_terrain_explorer()
                break
            else:
                print("❌ 无效选择，请输入 1-4")
        
        except KeyboardInterrupt:
            print("\n👋 程序退出")
            break
    
    print("\n✅ 预览完成!")
    print("💡 下一步: 运行 'python simple_terrain_viewer.py' 在Isaac Sim中查看3D效果")


if __name__ == "__main__":
    main()
