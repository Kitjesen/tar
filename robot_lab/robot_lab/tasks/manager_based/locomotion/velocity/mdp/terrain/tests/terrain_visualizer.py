"""
地形可视化器 - 在Isaac Lab中展示地形
用于测试和预览生成的地形效果
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional
# 尝试导入Isaac Lab相关模块
try:
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import Scene
    from isaaclab.utils import configclass
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim import SimulationContext
        from omni.isaac.lab.scene import Scene
        from omni.isaac.lab.utils import configclass
        ISAAC_LAB_AVAILABLE = True
    except ImportError:
        ISAAC_LAB_AVAILABLE = False
        print("Isaac Lab 不可用")

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


@configclass
class TerrainVisualizerCfg:
    """地形可视化器配置"""
    
    # 基础配置
    num_terrains: int = 4  # 展示的地形数量
    terrain_spacing: float = 12.0  # 地形间距
    
    # 地形大小
    terrain_size: tuple = (8.0, 8.0)
    resolution: float = 0.05  # 可视化用较粗分辨率
    
    # 材质配置
    terrain_material_path: str = "/World/Materials/TerrainMaterial"
    
    # 摄像机配置
    camera_distance: float = 20.0
    camera_height: float = 10.0
    

class TerrainVisualizer:
    """地形可视化器"""
    
    def __init__(self, cfg: TerrainVisualizerCfg):
        self.cfg = cfg
        self.sim = None
        self.scene = None
        self.terrain_prims = {}
        
    def create_heightfield_mesh(self, height_map: np.ndarray, 
                               resolution: float) -> tuple:
        """
        从高度图创建三角网格
        
        Args:
            height_map: 高度图数据
            resolution: 分辨率
            
        Returns:
            vertices: 顶点数组
            triangles: 三角面数组
        """
        rows, cols = height_map.shape
        
        # 创建顶点
        vertices = []
        for i in range(rows):
            for j in range(cols):
                x = i * resolution
                y = j * resolution
                z = height_map[i, j]
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # 创建三角面索引
        triangles = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                # 每个网格单元的四个顶点
                v1 = i * cols + j
                v2 = i * cols + (j + 1)
                v3 = (i + 1) * cols + j
                v4 = (i + 1) * cols + (j + 1)
                
                # 创建两个三角形
                triangles.extend([v1, v2, v3])  # 第一个三角形
                triangles.extend([v2, v4, v3])  # 第二个三角形
        
        triangles = np.array(triangles, dtype=np.int32)
        
        return vertices, triangles
    
    def create_terrain_prim(self, height_map: np.ndarray, 
                           prim_path: str, position: tuple = (0, 0, 0)) -> bool:
        """
        在Isaac Lab中创建地形Prim
        
        Args:
            height_map: 高度图
            prim_path: Prim路径
            position: 世界坐标位置
            
        Returns:
            success: 是否创建成功
        """
        try:
            import omni.usd
            from pxr import UsdGeom, Gf, UsdPhysics, Sdf
            
            # 获取当前stage
            stage = omni.usd.get_context().get_stage()
            
            # 创建网格数据
            vertices, triangles = self.create_heightfield_mesh(
                height_map, self.cfg.resolution
            )
            
            # 创建Mesh Prim
            mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
            
            # 设置顶点
            mesh_prim.CreatePointsAttr(vertices.tolist())
            
            # 设置面
            face_vertex_counts = [3] * (len(triangles) // 3)  # 每个面3个顶点
            mesh_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
            mesh_prim.CreateFaceVertexIndicesAttr(triangles.tolist())
            
            # 设置位置
            mesh_prim.AddTranslateOp().Set(Gf.Vec3f(*position))
            
            # 添加碰撞器
            collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
            
            # 添加网格碰撞形状
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
            mesh_collision_api.CreateApproximationAttr("none")  # 使用精确网格
            
            print(f"成功创建地形: {prim_path}")
            return True
            
        except Exception as e:
            print(f"创建地形失败 {prim_path}: {e}")
            return False
    
    def setup_scene(self):
        """设置Isaac Lab场景"""
        try:
            # 创建仿真配置
            sim_cfg = sim_utils.SimulationCfg(
                dt=1.0/60.0,
                device="cuda" if torch.cuda.is_available() else "cpu",
                gravity=(0.0, 0.0, -9.81),
                render_interval=1,
                enable_scene_query_support=True,
                use_fabric=True,
            )
            
            # 初始化仿真上下文
            self.sim = sim_utils.SimulationContext(sim_cfg)
            
            # 创建场景
            scene_cfg = sim_utils.SceneCfg(
                num_envs=1,
                env_spacing=0.0
            )
            self.scene = Scene(scene_cfg)
            
            print("Isaac Lab场景设置完成")
            return True
            
        except Exception as e:
            print(f"场景设置失败: {e}")
            return False
    
    def create_demo_terrains(self):
        """创建演示地形"""
        terrain_demos = []
        
        # 1. 崎岖地形
        rough_configs = create_rough_terrain_configs()
        rough_gen = RoughTerrainGenerator(rough_configs["medium"])
        rough_map = rough_gen.generate(seed=42)
        terrain_demos.append(("崎岖地形", rough_map))
        
        # 2. 上楼梯
        stairs_configs = create_stairs_terrain_configs()
        stairs_gen = StairsTerrainGenerator(stairs_configs["medium_up"])
        stairs_map = stairs_gen.generate(seed=42)
        terrain_demos.append(("上楼梯", stairs_map))
        
        # 3. 下楼梯
        stairs_down_gen = StairsTerrainGenerator(stairs_configs["medium_down"])
        stairs_down_map = stairs_down_gen.generate(seed=42)
        terrain_demos.append(("下楼梯", stairs_down_map))
        
        # 4. 跳跃障碍
        gap_configs = create_gap_terrain_configs()
        gap_gen = GapTerrainGenerator(gap_configs["medium_height"])
        gap_map = gap_gen.generate(seed=42)
        terrain_demos.append(("跳跃障碍", gap_map))
        
        return terrain_demos
    
    def visualize_terrains(self):
        """可视化所有地形"""
        if not self.setup_scene():
            return False
        
        # 创建演示地形
        terrain_demos = self.create_demo_terrains()
        
        # 在场景中创建地形
        for i, (name, height_map) in enumerate(terrain_demos):
            x_offset = (i % 2) * self.cfg.terrain_spacing
            y_offset = (i // 2) * self.cfg.terrain_spacing
            
            prim_path = f"/World/Terrain_{i}_{name.replace(' ', '_')}"
            position = (x_offset, y_offset, 0)
            
            success = self.create_terrain_prim(height_map, prim_path, position)
            
            if success:
                self.terrain_prims[name] = prim_path
                print(f"  {name}: 位置 ({x_offset:.1f}, {y_offset:.1f})")
        
        # 设置摄像机视角
        self._setup_camera()
        
        # 启动仿真
        self.sim.reset()
        print(f"\n成功创建 {len(self.terrain_prims)} 个地形")
        print("Isaac Lab可视化已启动！")
        
        return True
    
    def _setup_camera(self):
        """设置摄像机视角"""
        try:
            import omni.kit.viewport.utility as viewport_utils
            
            # 获取视口
            viewport = viewport_utils.get_active_viewport()
            if viewport:
                # 设置摄像机位置和目标
                center_x = self.cfg.terrain_spacing / 2
                center_y = self.cfg.terrain_spacing / 2
                
                camera_pos = (
                    center_x + self.cfg.camera_distance,
                    center_y + self.cfg.camera_distance,
                    self.cfg.camera_height
                )
                
                target_pos = (center_x, center_y, 0)
                
                # 这里需要根据具体的Isaac Lab版本调整摄像机设置方法
                print(f"摄像机位置: {camera_pos}")
                print(f"摄像机目标: {target_pos}")
                
        except Exception as e:
            print(f"摄像机设置失败: {e}")
    
    def run_interactive(self):
        """运行交互式可视化"""
        if not self.visualize_terrains():
            return
        
        print("\n=== 交互式地形可视化 ===")
        print("已创建的地形:")
        for i, (name, prim_path) in enumerate(self.terrain_prims.items()):
            print(f"  {i+1}. {name}: {prim_path}")
        
        print("\n提示:")
        print("- 使用鼠标拖拽旋转视角")
        print("- 使用滚轮缩放")
        print("- 按 ESC 退出")
        
        # 保持仿真运行
        try:
            while True:
                self.sim.step()
                # 这里可以添加实时更新逻辑
                
        except KeyboardInterrupt:
            print("\n可视化已停止")
    
    def export_terrain_info(self) -> Dict[str, Any]:
        """导出地形信息"""
        terrain_demos = self.create_demo_terrains()
        
        info = {
            "terrain_count": len(terrain_demos),
            "terrain_size": self.cfg.terrain_size,
            "resolution": self.cfg.resolution,
            "terrains": {}
        }
        
        for name, height_map in terrain_demos:
            info["terrains"][name] = {
                "shape": height_map.shape,
                "height_range": (float(np.min(height_map)), float(np.max(height_map))),
                "mean_height": float(np.mean(height_map)),
                "std_height": float(np.std(height_map)),
                "num_vertices": height_map.size,
                "num_triangles": (height_map.shape[0] - 1) * (height_map.shape[1] - 1) * 2
            }
        
        return info


def create_quick_test():
    """创建快速测试配置"""
    cfg = TerrainVisualizerCfg()
    cfg.num_terrains = 4
    cfg.terrain_spacing = 10.0
    cfg.terrain_size = (6.0, 6.0)
    cfg.resolution = 0.1
    
    return TerrainVisualizer(cfg)


if __name__ == "__main__":
    print("=== Isaac Lab 地形可视化器 ===")
    
    # 创建可视化器
    visualizer = create_quick_test()
    
    # 导出地形信息
    info = visualizer.export_terrain_info()
    print("地形信息:")
    print(f"  地形数量: {info['terrain_count']}")
    print(f"  地形大小: {info['terrain_size']}")
    print(f"  分辨率: {info['resolution']}")
    
    for name, terrain_info in info["terrains"].items():
        print(f"\n  {name}:")
        print(f"    网格大小: {terrain_info['shape']}")
        print(f"    高度范围: {terrain_info['height_range'][0]:.3f} ~ {terrain_info['height_range'][1]:.3f} m")
        print(f"    顶点数量: {terrain_info['num_vertices']}")
        print(f"    三角面数量: {terrain_info['num_triangles']}")
    
    print(f"\n准备启动Isaac Lab可视化...")
    print("注意: 需要在Isaac Lab环境中运行此脚本")
    
    # 如果在Isaac Lab环境中，启动可视化
    try:
        visualizer.run_interactive()
    except Exception as e:
        print(f"可视化启动失败: {e}")
        print("请确保在Isaac Lab环境中运行此脚本")
