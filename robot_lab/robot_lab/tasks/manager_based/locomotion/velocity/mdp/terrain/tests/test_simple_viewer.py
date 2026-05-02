#!/usr/bin/env python3

"""
简化版地形查看器测试
专门针对isaaclab版本优化
"""

import numpy as np

# 尝试导入Isaac Lab相关模块
try:
    from isaaclab.app import AppLauncher
    
    # 启动Isaac Lab应用
    app_launcher = AppLauncher({"headless": False})
    simulation_app = app_launcher.app
    
    import isaaclab.sim as sim_utils
    ISAAC_LAB_AVAILABLE = True
    print("✓ Isaac Lab (isaaclab) 导入成功")
    
except ImportError as e:
    print(f"❌ Isaac Lab 导入失败: {e}")
    print("请检查Isaac Lab安装")
    ISAAC_LAB_AVAILABLE = False

# 导入地形生成器 (从上级目录)
if ISAAC_LAB_AVAILABLE:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from rough_terrain import RoughTerrainGenerator, create_rough_terrain_configs
    from stairs_terrain import StairsTerrainGenerator, create_stairs_terrain_configs
    from gap_terrain import GapTerrainGenerator, create_gap_terrain_configs


def create_terrain_mesh(height_map: np.ndarray, resolution: float = 0.1):
    """将高度图转换为三角网格"""
    rows, cols = height_map.shape
    
    # 创建顶点
    vertices = []
    for i in range(rows):
        for j in range(cols):
            x = i * resolution - (rows * resolution) / 2  # 居中
            y = j * resolution - (cols * resolution) / 2  # 居中  
            z = height_map[i, j]
            vertices.append([x, y, z])
    
    # 创建三角面
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v1 = i * cols + j
            v2 = i * cols + (j + 1)
            v3 = (i + 1) * cols + j
            v4 = (i + 1) * cols + (j + 1)
            
            # 两个三角形
            faces.extend([v1, v2, v3])
            faces.extend([v2, v4, v3])
    
    return np.array(vertices), np.array(faces)


def create_terrain_prim(height_map: np.ndarray, prim_path: str, position: tuple, resolution: float = 0.1):
    """在Isaac Sim中创建地形"""
    try:
        import omni.usd
        from pxr import UsdGeom, Gf, UsdPhysics
        
        stage = omni.usd.get_context().get_stage()
        
        # 创建网格数据
        vertices, faces = create_terrain_mesh(height_map, resolution)
        
        # 创建网格Prim
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        
        # 设置顶点
        mesh.CreatePointsAttr(vertices.tolist())
        
        # 设置面（每个面3个顶点）
        face_vertex_counts = [3] * (len(faces) // 3)
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(faces.tolist())
        
        # 设置位置
        mesh.AddTranslateOp().Set(Gf.Vec3f(*position))
        
        # 添加物理碰撞
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        
        print(f"✓ 创建地形: {prim_path} 在位置 {position}")
        return True
        
    except Exception as e:
        print(f"❌ 创建地形失败 {prim_path}: {e}")
        return False


def setup_basic_scene():
    """设置基础场景"""
    try:
        # 添加地面
        ground_cfg = sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            color=(0.3, 0.3, 0.3)
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)
        
        # 添加光源
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.8, 0.8, 0.8)
        )
        light_cfg.func("/World/DomeLight", light_cfg)
        
        print("✓ 基础场景设置完成")
        return True
        
    except Exception as e:
        print(f"❌ 场景设置失败: {e}")
        return False


def create_test_terrains():
    """创建测试地形"""
    try:
        print("生成测试地形...")
        
        # 地形间距
        spacing = 8.0
        
        # 1. 简单崎岖地形
        print("1. 生成简单崎岖地形...")
        rough_configs = create_rough_terrain_configs()
        rough_gen = RoughTerrainGenerator(rough_configs["easy"])
        rough_map = rough_gen.generate(seed=42)
        
        success1 = create_terrain_prim(
            rough_map, 
            "/World/RoughTerrain", 
            (-spacing, -spacing, 0),
            resolution=0.08
        )
        
        # 2. 简单楼梯
        print("2. 生成简单楼梯...")
        stairs_configs = create_stairs_terrain_configs()
        stairs_gen = StairsTerrainGenerator(stairs_configs["easy_up"])
        stairs_map = stairs_gen.generate(seed=42)
        
        success2 = create_terrain_prim(
            stairs_map,
            "/World/StairsUp", 
            (spacing, -spacing, 0),
            resolution=0.08
        )
        
        # 3. 简单间隙
        print("3. 生成简单跳跃障碍...")
        gap_configs = create_gap_terrain_configs()
        gap_gen = GapTerrainGenerator(gap_configs["easy_simple"])
        gap_map = gap_gen.generate(seed=42)
        
        success3 = create_terrain_prim(
            gap_map,
            "/World/GapTerrain",
            (-spacing, spacing, 0),
            resolution=0.08
        )
        
        # 4. 平面（对比）
        print("4. 创建平面参考...")
        flat_map = np.zeros((100, 100))
        
        success4 = create_terrain_prim(
            flat_map,
            "/World/FlatTerrain",
            (spacing, spacing, 0),
            resolution=0.08
        )
        
        success_count = sum([success1, success2, success3, success4])
        print(f"✓ 成功创建 {success_count}/4 个地形")
        
        if success_count > 0:
            print("\n地形布局:")
            print(f"  左下: 崎岖地形   右下: 上楼梯")
            print(f"  左上: 跳跃障碍   右上: 平面参考")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 地形创建失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀" + "=" * 48 + "🚀")
    print("    Isaac Lab 地形查看器测试版")
    print("🚀" + "=" * 48 + "🚀")
    
    if not ISAAC_LAB_AVAILABLE:
        print("\n❌ Isaac Lab 不可用")
        print("请运行: python quick_preview.py")
        return
    
    try:
        # 创建仿真配置（简化版）
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0/60.0,
            device="cpu",  # 使用CPU避免CUDA问题
            gravity=(0.0, 0.0, -9.81),
            render_interval=1,
        )
        
        # 初始化仿真
        print("初始化仿真...")
        sim_context = sim_utils.SimulationContext(sim_cfg)
        
        # 设置摄像机
        sim_context.set_camera_view([20.0, 20.0, 15.0], [0.0, 0.0, 0.0])
        
        # 设置场景
        if not setup_basic_scene():
            return
        
        # 创建地形
        if not create_test_terrains():
            return
        
        # 重置仿真
        sim_context.reset()
        
        print("\n" + "🎉" * 20)
        print("Isaac Sim 地形查看器启动成功!")
        print("🎉" * 20)
        
        print("\n操作说明:")
        print("🖱️  左键拖拽: 旋转视角")
        print("🖱️  中键拖拽: 平移") 
        print("🖱️  滚轮: 缩放")
        print("⌨️  ESC: 退出")
        
        # 仿真循环
        frame_count = 0
        try:
            while simulation_app.is_running():
                sim_context.step()
                
                if frame_count % 300 == 0 and frame_count > 0:
                    print(f"⏱️  运行中... (帧: {frame_count})")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n🛑 用户中断")
        
        print("👋 程序结束")
        simulation_app.close()
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 故障排除:")
        print("1. 确认Isaac Lab正确安装")
        print("2. 检查虚拟环境激活")
        print("3. 运行: python quick_preview.py (matplotlib版本)")


if __name__ == "__main__":
    main()
