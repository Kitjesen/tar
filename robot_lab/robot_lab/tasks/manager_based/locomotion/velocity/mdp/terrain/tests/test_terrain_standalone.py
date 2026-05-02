"""
独立地形测试脚本 - 不依赖Isaac Lab
用于快速测试地形生成功能和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import os

# 导入地形模块
try:
    from . import (
        TerrainManager, create_terrain_manager_configs,
        RoughTerrainGenerator, create_rough_terrain_configs,
        StairsTerrainGenerator, create_stairs_terrain_configs, StairsDirection,
        GapTerrainGenerator, create_gap_terrain_configs, GapType
    )
except ImportError:
    from terrain_manager import TerrainManager, create_terrain_manager_configs
    from rough_terrain import RoughTerrainGenerator, create_rough_terrain_configs
    from stairs_terrain import StairsTerrainGenerator, create_stairs_terrain_configs, StairsDirection
    from gap_terrain import GapTerrainGenerator, create_gap_terrain_configs, GapType


def visualize_height_map(height_map: np.ndarray, title: str = "地形", 
                        save_path: str = None, show: bool = True):
    """
    可视化高度图
    
    Args:
        height_map: 高度图数据
        title: 图像标题
        save_path: 保存路径
        show: 是否显示图像
    """
    plt.figure(figsize=(10, 8))
    
    # 使用terrain colormap显示高度图
    im = plt.imshow(height_map.T, origin='lower', cmap='terrain', aspect='equal')
    plt.colorbar(im, label='高度 (m)', shrink=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X 方向 (网格点)', fontsize=12)
    plt.ylabel('Y 方向 (网格点)', fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    min_h, max_h = np.min(height_map), np.max(height_map)
    mean_h, std_h = np.mean(height_map), np.std(height_map)
    
    info_text = f"""统计信息:
高度范围: {min_h:.3f} ~ {max_h:.3f} m
平均高度: {mean_h:.3f} m
标准差: {std_h:.3f} m
网格大小: {height_map.shape[0]} × {height_map.shape[1]}"""
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def test_individual_terrains():
    """测试单个地形生成器"""
    print("=== 测试单个地形生成器 ===")
    
    output_dir = "terrain_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 测试崎岖地形
    print("\n1. 测试崎岖地形...")
    rough_configs = create_rough_terrain_configs()
    
    for difficulty, config in rough_configs.items():
        print(f"  生成 {difficulty} 难度崎岖地形...")
        generator = RoughTerrainGenerator(config)
        height_map = generator.generate(seed=42)
        
        title = f"崎岖地形 - {difficulty}"
        save_path = os.path.join(output_dir, f"rough_{difficulty}.png")
        visualize_height_map(height_map, title, save_path, show=False)
    
    # 2. 测试楼梯地形
    print("\n2. 测试楼梯地形...")
    stairs_configs = create_stairs_terrain_configs()
    
    for name, config in stairs_configs.items():
        print(f"  生成 {name} 楼梯地形...")
        generator = StairsTerrainGenerator(config)
        height_map = generator.generate(seed=42)
        
        title = f"楼梯地形 - {name}"
        save_path = os.path.join(output_dir, f"stairs_{name}.png")
        visualize_height_map(height_map, title, save_path, show=False)
    
    # 3. 测试跳跃障碍
    print("\n3. 测试跳跃障碍...")
    gap_configs = create_gap_terrain_configs()
    
    for name, config in gap_configs.items():
        print(f"  生成 {name} 跳跃障碍...")
        generator = GapTerrainGenerator(config)
        height_map = generator.generate(seed=42)
        
        title = f"跳跃障碍 - {name}"
        save_path = os.path.join(output_dir, f"gap_{name}.png")
        visualize_height_map(height_map, title, save_path, show=False)
    
    print(f"\n所有地形图像已保存到: {output_dir}/")


def test_terrain_manager():
    """测试地形管理器"""
    print("\n=== 测试地形管理器 ===")
    
    # 创建小规模测试配置
    manager_configs = create_terrain_manager_configs()
    config = manager_configs["balanced"]
    config.num_envs = 9  # 3x3网格便于可视化
    
    manager = TerrainManager(config)
    
    # 创建地形组合可视化
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for env_id in range(config.num_envs):
        height_map = manager.get_terrain_height_map(env_id)
        info = manager.get_terrain_info(env_id)
        
        # 可视化每个环境的地形
        im = axes[env_id].imshow(height_map.T, origin='lower', cmap='terrain', aspect='equal')
        
        title = f"环境 {env_id}\n{info['terrain_type']} (难度 {info['difficulty_level']})"
        axes[env_id].set_title(title, fontsize=10)
        axes[env_id].set_xlabel('X', fontsize=8)
        axes[env_id].set_ylabel('Y', fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[env_id], shrink=0.8)
        cbar.set_label('高度 (m)', fontsize=8)
    
    plt.suptitle("地形管理器 - 环境地形分配", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = "terrain_outputs/terrain_manager_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"地形管理器概览已保存到: {output_path}")
    plt.show()
    
    # 打印统计信息
    stats = manager.get_curriculum_statistics()
    print(f"\n地形管理器统计:")
    print(f"  环境数量: {stats['total_environments']}")
    print(f"  平均难度: {stats['average_difficulty']:.2f}")
    print(f"  地形分布: {stats['terrain_type_distribution']}")
    print(f"  难度分布: {stats['difficulty_distribution']}")


def test_curriculum_progression():
    """测试课程学习进展"""
    print("\n=== 测试课程学习进展 ===")
    
    # 测试单个地形类型的课程进展
    terrain_types = [
        ("崎岖地形", RoughTerrainGenerator, create_rough_terrain_configs()["medium"]),
        ("楼梯地形", StairsTerrainGenerator, create_stairs_terrain_configs()["medium_up"]),
        ("跳跃障碍", GapTerrainGenerator, create_gap_terrain_configs()["medium_height"])
    ]
    
    for terrain_name, generator_class, config in terrain_types:
        print(f"\n测试 {terrain_name} 课程进展...")
        
        generator = generator_class(config)
        curriculum = generator.generate_curriculum(num_levels=5)
        
        # 创建课程进展可视化
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        for level, height_map in curriculum.items():
            im = axes[level].imshow(height_map.T, origin='lower', cmap='terrain', aspect='equal')
            axes[level].set_title(f"难度等级 {level}", fontsize=12)
            axes[level].set_xlabel('X', fontsize=10)
            axes[level].set_ylabel('Y', fontsize=10)
            
            # 添加统计信息
            max_h = np.max(height_map)
            std_h = np.std(height_map)
            axes[level].text(0.02, 0.98, f"最大: {max_h:.3f}m\n标准差: {std_h:.3f}m", 
                           transform=axes[level].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=8)
            
            plt.colorbar(im, ax=axes[level], shrink=0.8)
        
        plt.suptitle(f"{terrain_name} - 课程学习进展", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = f"terrain_outputs/{terrain_name}_curriculum.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  {terrain_name} 课程进展已保存到: {output_path}")
        plt.show()


def generate_comparison_report():
    """生成地形对比报告"""
    print("\n=== 生成地形对比报告 ===")
    
    # 生成标准化的地形进行对比
    terrain_size = (6.0, 6.0)
    resolution = 0.05
    
    terrains = {}
    
    # 崎岖地形
    from rough_terrain import RoughTerrainConfig
    rough_config = RoughTerrainConfig(terrain_size=terrain_size, resolution=resolution, difficulty_level=2)
    rough_gen = RoughTerrainGenerator(rough_config)
    terrains["崎岖地形"] = rough_gen.generate(seed=42)
    
    # 上楼梯
    from stairs_terrain import StairsTerrainConfig
    stairs_config = StairsTerrainConfig(
        terrain_size=terrain_size, 
        resolution=resolution, 
        direction=StairsDirection.UP,
        difficulty_level=2
    )
    stairs_gen = StairsTerrainGenerator(stairs_config)
    terrains["上楼梯"] = stairs_gen.generate(seed=42)
    
    # 下楼梯
    stairs_down_config = StairsTerrainConfig(
        terrain_size=terrain_size,
        resolution=resolution,
        direction=StairsDirection.DOWN,
        difficulty_level=2
    )
    stairs_down_gen = StairsTerrainGenerator(stairs_down_config)
    terrains["下楼梯"] = stairs_down_gen.generate(seed=42)
    
    # 跳跃障碍
    from gap_terrain import GapTerrainConfig
    gap_config = GapTerrainConfig(
        terrain_size=terrain_size,
        resolution=resolution,
        gap_type=GapType.HEIGHT_GAP,
        difficulty_level=2
    )
    gap_gen = GapTerrainGenerator(gap_config)
    terrains["跳跃障碍"] = gap_gen.generate(seed=42)
    
    # 创建对比可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (name, height_map) in enumerate(terrains.items()):
        im = axes[i].imshow(height_map.T, origin='lower', cmap='terrain', aspect='equal')
        axes[i].set_title(f"{name}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel('X 方向', fontsize=12)
        axes[i].set_ylabel('Y 方向', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_label('高度 (m)', fontsize=10)
        
        # 添加统计信息
        min_h, max_h = np.min(height_map), np.max(height_map)
        mean_h, std_h = np.mean(height_map), np.std(height_map)
        
        info_text = f"""高度: {min_h:.3f}~{max_h:.3f}m
平均: {mean_h:.3f}m
标准差: {std_h:.3f}m"""
        
        axes[i].text(0.02, 0.98, info_text, transform=axes[i].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='white', alpha=0.8), fontsize=9)
    
    plt.suptitle("地形类型对比 - 中等难度", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = "terrain_outputs/terrain_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"地形对比图已保存到: {output_path}")
    plt.show()
    
    # 生成数据表格
    print("\n地形特征对比表:")
    print(f"{'地形类型':<10} {'最小高度':<8} {'最大高度':<8} {'平均高度':<8} {'标准差':<8} {'复杂度':<8}")
    print("-" * 60)
    
    for name, height_map in terrains.items():
        min_h = np.min(height_map)
        max_h = np.max(height_map)
        mean_h = np.mean(height_map)
        std_h = np.std(height_map)
        complexity = std_h / (max_h - min_h + 1e-6)  # 复杂度指标
        
        print(f"{name:<10} {min_h:<8.3f} {max_h:<8.3f} {mean_h:<8.3f} {std_h:<8.3f} {complexity:<8.3f}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("机器狗地形系统 - 完整测试")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs("terrain_outputs", exist_ok=True)
    
    try:
        # 1. 测试单个地形生成器
        test_individual_terrains()
        
        # 2. 测试地形管理器
        test_terrain_manager()
        
        # 3. 测试课程学习
        test_curriculum_progression()
        
        # 4. 生成对比报告
        generate_comparison_report()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
        print("\n生成的文件:")
        output_dir = "terrain_outputs"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith('.png'):
                    print(f"  - {file}")
        
        print(f"\n所有测试结果已保存到: {output_dir}/")
        print("\n下一步:")
        print("1. 查看生成的图像了解地形效果")
        print("2. 运行 terrain_visualizer.py 在Isaac Lab中查看3D效果")
        print("3. 集成到你的训练环境中")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
