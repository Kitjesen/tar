"""
地形模块使用示例
演示如何使用各种地形生成器和管理器
"""

import numpy as np
from typing import Dict, Any

# 导入地形模块
from . import (
    RoughTerrainGenerator, create_rough_terrain_configs,
    StairsTerrainGenerator, create_stairs_terrain_configs, StairsDirection,
    GapTerrainGenerator, create_gap_terrain_configs, GapType,
    TerrainManager, create_terrain_manager_configs, CurriculumStrategy
)


def demo_rough_terrain():
    """演示崎岖地形生成"""
    print("=== 崎岖地形演示 ===")
    
    # 获取预设配置
    configs = create_rough_terrain_configs()
    
    for difficulty, config in configs.items():
        print(f"\n{difficulty} 难度崎岖地形:")
        
        # 创建生成器
        generator = RoughTerrainGenerator(config)
        
        # 生成地形
        height_map = generator.generate(seed=42)
        
        # 显示统计信息
        print(f"  地形大小: {config.terrain_size}")
        print(f"  网格尺寸: {generator.grid_size}")
        print(f"  高度范围: {np.min(height_map):.3f} ~ {np.max(height_map):.3f} m")
        print(f"  平均高度: {np.mean(height_map):.3f} m")
        print(f"  高度标准差: {np.std(height_map):.3f} m")
        
        # 获取地形信息
        info = generator.get_terrain_info()
        print(f"  最大高度: {info['max_height']:.3f} m")
        print(f"  最大坡度: {info['max_slope']:.1f} 度")


def demo_stairs_terrain():
    """演示楼梯地形生成"""
    print("\n=== 楼梯地形演示 ===")
    
    # 获取预设配置
    configs = create_stairs_terrain_configs()
    
    for name, config in configs.items():
        print(f"\n{name} 楼梯配置:")
        
        # 创建生成器
        generator = StairsTerrainGenerator(config)
        
        # 生成地形
        height_map = generator.generate(seed=42)
        
        # 显示统计信息
        print(f"  方向: {config.direction.value}")
        print(f"  地形大小: {config.terrain_size}")
        print(f"  高度范围: {np.min(height_map):.3f} ~ {np.max(height_map):.3f} m")
        print(f"  台阶高度范围: {config.step_height}")
        print(f"  台阶数量范围: {config.num_steps}")
        
        # 获取地形信息
        info = generator.get_terrain_info()
        print(f"  当前台阶高度范围: {info['step_height_range']}")
        print(f"  当前台阶数量范围: {info['steps_range']}")


def demo_gap_terrain():
    """演示跳跃障碍地形生成"""
    print("\n=== 跳跃障碍地形演示 ===")
    
    # 获取预设配置
    configs = create_gap_terrain_configs()
    
    for name, config in configs.items():
        print(f"\n{name} 障碍配置:")
        
        # 创建生成器
        generator = GapTerrainGenerator(config)
        
        # 生成地形
        height_map = generator.generate(seed=42)
        
        # 显示统计信息
        print(f"  间隙类型: {config.gap_type.value}")
        print(f"  地形大小: {config.terrain_size}")
        print(f"  高度范围: {np.min(height_map):.3f} ~ {np.max(height_map):.3f} m")
        print(f"  间隙宽度范围: {config.gap_width}")
        print(f"  障碍数量范围: {config.num_obstacles}")
        
        # 获取地形信息
        info = generator.get_terrain_info()
        print(f"  当前间隙宽度范围: {info['gap_width_range']}")
        print(f"  当前障碍数量范围: {info['obstacles_range']}")


def demo_terrain_manager():
    """演示地形管理器"""
    print("\n=== 地形管理器演示 ===")
    
    # 获取预设配置
    configs = create_terrain_manager_configs()
    
    for name, config in configs.items():
        print(f"\n{name} 管理器配置:")
        
        # 创建地形管理器
        manager = TerrainManager(config)
        
        # 显示基本信息
        print(f"  环境数量: {config.num_envs}")
        print(f"  地形权重: {config.terrain_weights}")
        print(f"  课程策略: {config.curriculum_strategy.value}")
        print(f"  成功率阈值: {config.success_rate_threshold}")
        
        # 显示环境分配示例
        print(f"\n  前5个环境分配:")
        for env_id in range(min(5, config.num_envs)):
            info = manager.get_terrain_info(env_id)
            pos = manager.get_env_position(env_id)
            print(f"    环境 {env_id}: {info['terrain_type']} (难度 {info['difficulty_level']}) @ ({pos[0]:.1f}, {pos[1]:.1f})")
        
        # 获取统计信息
        stats = manager.get_curriculum_statistics()
        print(f"\n  统计信息:")
        print(f"    难度分布: {stats['difficulty_distribution']}")
        print(f"    地形分布: {stats['terrain_type_distribution']}")
        print(f"    平均难度: {stats['average_difficulty']:.2f}")


def demo_curriculum_learning():
    """演示课程学习"""
    print("\n=== 课程学习演示 ===")
    
    # 创建自适应管理器
    config = create_terrain_manager_configs()["balanced"]
    config.num_envs = 16  # 使用较少环境便于演示
    manager = TerrainManager(config)
    
    print(f"初始状态:")
    stats = manager.get_curriculum_statistics()
    print(f"  平均难度: {stats['average_difficulty']:.2f}")
    print(f"  难度分布: {stats['difficulty_distribution']}")
    
    # 模拟训练过程
    for episode in range(5):
        print(f"\n第 {episode + 1} 轮训练:")
        
        # 模拟每个环境的成功率
        for env_id in range(config.num_envs):
            # 高成功率环境应该升级难度
            if env_id < config.num_envs // 2:
                success_rate = np.random.uniform(0.85, 0.95)  # 高成功率
            else:
                success_rate = np.random.uniform(0.5, 0.7)   # 低成功率
                
            manager.update_success_rate(env_id, success_rate)
        
        # 更新课程
        changed_envs = manager.update_curriculum()
        
        # 显示结果
        stats = manager.get_curriculum_statistics()
        print(f"  难度变化环境: {len(changed_envs)} 个")
        print(f"  新平均难度: {stats['average_difficulty']:.2f}")
        print(f"  平均成功率: {stats['average_success_rate']:.3f}")


def demo_terrain_generation_comparison():
    """对比不同地形生成效果"""
    print("\n=== 地形生成对比 ===")
    
    terrain_size = (6.0, 6.0)
    resolution = 0.03
    
    # 崎岖地形
    from .rough_terrain import RoughTerrainConfig
    rough_config = RoughTerrainConfig(terrain_size=terrain_size, resolution=resolution, difficulty_level=2)
    rough_gen = RoughTerrainGenerator(rough_config)
    rough_map = rough_gen.generate(seed=42)
    
    # 楼梯地形
    from .stairs_terrain import StairsTerrainConfig
    stairs_config = StairsTerrainConfig(
        terrain_size=terrain_size, 
        resolution=resolution, 
        direction=StairsDirection.UP,
        difficulty_level=2
    )
    stairs_gen = StairsTerrainGenerator(stairs_config)
    stairs_map = stairs_gen.generate(seed=42)
    
    # 跳跃障碍
    from .gap_terrain import GapTerrainConfig
    gap_config = GapTerrainConfig(
        terrain_size=terrain_size,
        resolution=resolution,
        gap_type=GapType.HEIGHT_GAP,
        difficulty_level=2
    )
    gap_gen = GapTerrainGenerator(gap_config)
    gap_map = gap_gen.generate(seed=42)
    
    # 对比统计
    terrains = {
        "崎岖地形": rough_map,
        "楼梯地形": stairs_map, 
        "跳跃障碍": gap_map,
    }
    
    print("地形特征对比:")
    print(f"{'地形类型':<10} {'最小高度':<8} {'最大高度':<8} {'平均高度':<8} {'标准差':<8}")
    print("-" * 50)
    
    for name, height_map in terrains.items():
        min_h = np.min(height_map)
        max_h = np.max(height_map)
        mean_h = np.mean(height_map)
        std_h = np.std(height_map)
        print(f"{name:<10} {min_h:<8.3f} {max_h:<8.3f} {mean_h:<8.3f} {std_h:<8.3f}")


def main():
    """主演示函数"""
    print("机器狗地形系统演示")
    print("=" * 50)
    
    try:
        # 演示各种地形生成器
        demo_rough_terrain()
        demo_stairs_terrain()
        demo_gap_terrain()
        
        # 演示地形管理器
        demo_terrain_manager()
        
        # 演示课程学习
        demo_curriculum_learning()
        
        # 演示地形对比
        demo_terrain_generation_comparison()
        
        print("\n" + "=" * 50)
        print("演示完成!")
        
        print("\n使用说明:")
        print("1. 导入所需的地形生成器类")
        print("2. 创建配置或使用预设配置")
        print("3. 实例化生成器并调用generate()方法")
        print("4. 使用TerrainManager统一管理多个环境的地形")
        print("5. 通过update_curriculum()实现自动难度调整")
        
        print("\n主要文件:")
        print("- rough_terrain.py: 崎岖地形生成")
        print("- stairs_terrain.py: 楼梯地形生成")
        print("- gap_terrain.py: 跳跃障碍生成")
        print("- terrain_manager.py: 统一地形管理")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
