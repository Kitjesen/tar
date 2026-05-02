#!/usr/bin/env python3


import numpy as np

# 尝试导入Isaac Lab相关模块
try:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": False})
    simulation_app = app_launcher.app
    
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    ISAAC_LAB_AVAILABLE = True
    print("✓ 使用 isaaclab 模块")
    
except ImportError:
    try:
        from omni.isaac.lab.app import AppLauncher
        app_launcher = AppLauncher({"headless": False})
        simulation_app = app_launcher.app
        
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim import SimulationContext
        ISAAC_LAB_AVAILABLE = True
        print("✓ 使用 omni.isaac.lab 模块")
        
    except ImportError:
        print("❌ Isaac Lab 未安装或配置错误")
        print("请检查:")
        print("1. Isaac Lab是否正确安装")
        print("2. 虚拟环境是否激活")
        print("3. PYTHONPATH是否正确设置")
        ISAAC_LAB_AVAILABLE = False

# 导入我们的地形生成器 (从上级目录)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rough_terrain import RoughTerrainGenerator, RoughTerrainConfig, create_rough_terrain_configs
from stairs_terrain import StairsTerrainGenerator, StairsTerrainConfig, create_stairs_terrain_configs
from gap_terrain import GapTerrainGenerator, GapTerrainConfig, create_gap_terrain_configs


def create_mesh_from_heightmap(height_map: np.ndarray, resolution: float = 0.1):
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
            # 每个网格单元的四个顶点索引
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
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics
    
    stage = omni.usd.get_context().get_stage()
    
    # 创建网格数据
    vertices, faces = create_mesh_from_heightmap(height_map, resolution)
    
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


def setup_scene():
    """设置场景"""
    # 添加地面
    ground_cfg = sim_utils.GroundPlaneCfg(
        size=(50.0, 50.0),
        color=(0.2, 0.2, 0.2)  # 深灰色地面
    )
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    
    # 添加圆顶光源（更适合地形显示）
    light_cfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75)
    )
    light_cfg.func("/World/Light", light_cfg)
    
    print("✓ 场景设置完成")


def create_demo_terrains():
    """创建演示地形"""
    print("开始生成地形...")
    
    # 地形间距
    spacing = 10.0
    
    # 1. 崎岖地形 - 中等难度
    print("1. 生成崎岖地形...")
    rough_configs = create_rough_terrain_configs()
    rough_gen = RoughTerrainGenerator(rough_configs["medium"])
    rough_map = rough_gen.generate(seed=42)
    
    create_terrain_prim(
        rough_map, 
        "/World/RoughTerrain", 
        (-spacing, -spacing, 0),
        resolution=0.05
    )
    
    # 2. 上楼梯
    print("2. 生成上楼梯...")
    stairs_configs = create_stairs_terrain_configs()
    stairs_gen = StairsTerrainGenerator(stairs_configs["medium_up"])
    stairs_map = stairs_gen.generate(seed=42)
    
    create_terrain_prim(
        stairs_map,
        "/World/StairsUp", 
        (spacing, -spacing, 0),
        resolution=0.05
    )
    
    # 3. 下楼梯
    print("3. 生成下楼梯...")
    stairs_down_gen = StairsTerrainGenerator(stairs_configs["medium_down"])  
    stairs_down_map = stairs_down_gen.generate(seed=42)
    
    create_terrain_prim(
        stairs_down_map,
        "/World/StairsDown",
        (-spacing, spacing, 0), 
        resolution=0.05
    )
    
    # 4. 跳跃障碍
    print("4. 生成跳跃障碍...")
    gap_configs = create_gap_terrain_configs()
    gap_gen = GapTerrainGenerator(gap_configs["medium_height"])
    gap_map = gap_gen.generate(seed=42)
    
    create_terrain_prim(
        gap_map,
        "/World/GapTerrain",
        (spacing, spacing, 0),
        resolution=0.05
    )
    
    print("✓ 所有地形创建完成!")
    print("\n地形布局:")
    print(f"  崎岖地形:   位置 ({-spacing}, {-spacing})")
    print(f"  上楼梯:     位置 ({spacing}, {-spacing})")
    print(f"  下楼梯:     位置 ({-spacing}, {spacing})")
    print(f"  跳跃障碍:   位置 ({spacing}, {spacing})")


def main():
    """主函数"""
    print("=" * 50)
    print("Isaac Sim 地形可视化器")
    print("=" * 50)
    
    if not ISAAC_LAB_AVAILABLE:
        print("\n❌ Isaac Lab 不可用，请运行以下命令:")
        print("   python quick_preview.py  # 使用matplotlib查看地形")
        return
    
    try:
        # 创建仿真配置
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0/60.0,
            device="cuda" if ISAAC_LAB_AVAILABLE else "cpu",
            gravity=(0.0, 0.0, -9.81),
            render_interval=1,
            enable_scene_query_support=True,
            use_fabric=True,
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                min_position_iteration_count=4,
                max_position_iteration_count=4,
            ),
        )
        
        # 初始化仿真上下文
        sim_context = sim_utils.SimulationContext(sim_cfg)
        
        # 设置摄像机视角
        sim_context.set_camera_view([15.0, 15.0, 10.0], [0.0, 0.0, 0.0])
        
        # 设置场景
        setup_scene()
        
        # 创建地形
        create_demo_terrains()
        
        # 重置仿真
        sim_context.reset()
        
        print("\n" + "=" * 50)
        print("Isaac Sim 已启动!")
        print("=" * 50)
        print("\n操作指南:")
        print("🖱️  鼠标左键拖拽: 旋转视角")
        print("🖱️  鼠标中键拖拽: 平移视角")  
        print("🖱️  鼠标滚轮: 缩放")
        print("⌨️  W/A/S/D: 移动摄像机")
        print("⌨️  Q/E: 上下移动摄像机")
        print("⌨️  ESC: 退出")
        
        print("\n地形说明:")
        print("🏔️  左下角: 崎岖地形 (随机高度变化)")
        print("🪜  右下角: 上楼梯 (台阶向上)")
        print("🪜  左上角: 下楼梯 (台阶向下)")  
        print("🕳️  右上角: 跳跃障碍 (间隙和平台)")
        
        # 运行仿真循环
        frame_count = 0
        try:
            while simulation_app.is_running():
                sim_context.step()
                
                # 每5秒打印一次提示
                if frame_count % 300 == 0 and frame_count > 0:
                    print(f"⏱️  仿真运行中... (帧数: {frame_count})")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n🛑 接收到退出信号...")
        
        print("👋 程序结束")
        simulation_app.close()
        
    except Exception as e:
        print(f"❌ Isaac Lab 启动失败: {e}")
        print("\n💡 建议:")
        print("1. 检查 Isaac Lab 安装: pip list | grep isaac")
        print("2. 确认虚拟环境: conda info --envs")  
        print("3. 运行简单版本: python quick_preview.py")


if __name__ == "__main__":
    main()
