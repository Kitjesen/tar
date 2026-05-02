"""
Isaac Lab 地形演示脚本
在Isaac Lab中创建并展示地形的完整示例
"""

import torch
import numpy as np

# 尝试导入AppLauncher
try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

# 启动应用
config = {"headless": False}  # 设置为False以显示GUI
app_launcher = AppLauncher(config)
simulation_app = app_launcher.app

# 导入Isaac Lab模块
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
from terrain_manager import TerrainManager, create_terrain_manager_configs
from rough_terrain import RoughTerrainGenerator, create_rough_terrain_configs
from stairs_terrain import StairsTerrainGenerator, create_stairs_terrain_configs
from gap_terrain import GapTerrainGenerator, create_gap_terrain_configs


@configclass
class TerrainDemoEnvCfg:
    """地形演示环境配置"""
    
    # 仿真配置
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1/60,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=512 * 1024,
            gpu_max_rigid_patch_count=32 * 1024,
        )
    )
    
    # 场景配置
    scene: sim_utils.SceneCfg = sim_utils.SceneCfg(
        num_envs=1,
        env_spacing=0.0
    )
    
    # 地形配置
    terrain_spacing: float = 12.0
    terrain_size: tuple = (8.0, 8.0)
    resolution: float = 0.1


class TerrainDemoEnv:
    """地形演示环境"""
    
    def __init__(self, cfg: TerrainDemoEnvCfg):
        self.cfg = cfg
        self.sim = None
        self.scene = None
        self.terrain_meshes = {}
        
    def _create_heightfield_terrain(self, height_map: np.ndarray, 
                                   prim_path: str, position: tuple = (0, 0, 0)):
        """
        创建高度场地形
        
        Args:
            height_map: 高度图数据
            prim_path: Prim路径
            position: 世界坐标位置
        """
        # 使用Isaac Lab的地形创建工具
        terrain_cfg = sim_utils.GroundPlaneCfg(
            prim_path=prim_path,
            size=(self.cfg.terrain_size[0], self.cfg.terrain_size[1]),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        )
        
        # 这里需要根据实际的Isaac Lab API创建高度场
        # 以下是伪代码，需要根据具体版本调整
        try:
            # 方法1: 使用TerrainImporter (如果可用)
            from omni.isaac.lab.terrains import TerrainImporterCfg
            
            # 将高度图转换为网格
            vertices, triangles = self._heightmap_to_mesh(height_map)
            
            terrain_importer_cfg = TerrainImporterCfg(
                prim_path=prim_path,
                terrain_type="mesh",
                mesh_vertices=vertices,
                mesh_triangles=triangles,
                physics_material=terrain_cfg.physics_material,
            )
            
            # 创建地形
            terrain_importer_cfg.func(prim_path, terrain_importer_cfg)
            
        except ImportError:
            # 方法2: 使用基础几何体近似
            print(f"使用近似方法创建地形: {prim_path}")
            self._create_approximated_terrain(height_map, prim_path, position)
    
    def _heightmap_to_mesh(self, height_map: np.ndarray):
        """将高度图转换为三角网格"""
        rows, cols = height_map.shape
        
        # 创建顶点
        vertices = []
        for i in range(rows):
            for j in range(cols):
                x = i * self.cfg.resolution
                y = j * self.cfg.resolution
                z = height_map[i, j]
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # 创建三角面
        triangles = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v1 = i * cols + j
                v2 = i * cols + (j + 1)
                v3 = (i + 1) * cols + j
                v4 = (i + 1) * cols + (j + 1)
                
                triangles.extend([v1, v2, v3])
                triangles.extend([v2, v4, v3])
        
        triangles = np.array(triangles, dtype=np.int32)
        return vertices, triangles
    
    def _create_approximated_terrain(self, height_map: np.ndarray, 
                                   prim_path: str, position: tuple):
        """创建近似地形（使用简单几何体）"""
        # 这是一个简化的实现，实际使用时应该用真正的高度场
        
        # 计算平均高度作为地面高度
        avg_height = np.mean(height_map)
        
        # 创建基础地面
        ground_cfg = sim_utils.GroundPlaneCfg(
            prim_path=prim_path,
            size=self.cfg.terrain_size,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        )
        
        # 添加到场景
        ground_cfg.func(prim_path, ground_cfg)
        
        print(f"创建近似地形: {prim_path} (平均高度: {avg_height:.3f}m)")
    
    def _setup_scene(self):
        """设置场景"""
        # 创建地面
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        
        # 创建演示地形
        self._create_demo_terrains()
        
        # 添加光源
        light_cfg = sim_utils.DistantLightCfg(
            prim_path="/World/light",
            color=(1.0, 1.0, 1.0),
            intensity=1000.0
        )
        light_cfg.func("/World/light", light_cfg)
    
    def _create_demo_terrains(self):
        """创建演示地形"""
        # 1. 崎岖地形
        rough_configs = create_rough_terrain_configs()
        rough_gen = RoughTerrainGenerator(rough_configs["medium"])
        rough_map = rough_gen.generate(seed=42)
        
        self._create_heightfield_terrain(
            rough_map, 
            "/World/Terrain_Rough", 
            (0, 0, 0)
        )
        
        # 2. 楼梯地形
        stairs_configs = create_stairs_terrain_configs()
        stairs_gen = StairsTerrainGenerator(stairs_configs["medium_up"])
        stairs_map = stairs_gen.generate(seed=42)
        
        self._create_heightfield_terrain(
            stairs_map,
            "/World/Terrain_Stairs",
            (self.cfg.terrain_spacing, 0, 0)
        )
        
        # 3. 跳跃障碍
        gap_configs = create_gap_terrain_configs()
        gap_gen = GapTerrainGenerator(gap_configs["medium_height"])
        gap_map = gap_gen.generate(seed=42)
        
        self._create_heightfield_terrain(
            gap_map,
            "/World/Terrain_Gaps",
            (0, self.cfg.terrain_spacing, 0)
        )
        
        # 4. 混合地形（使用地形管理器）
        manager_config = create_terrain_manager_configs()["balanced"]
        manager_config.num_envs = 1
        manager = TerrainManager(manager_config)
        
        mixed_map = manager.get_terrain_height_map(0)
        self._create_heightfield_terrain(
            mixed_map,
            "/World/Terrain_Mixed",
            (self.cfg.terrain_spacing, self.cfg.terrain_spacing, 0)
        )
        
        print("已创建4个演示地形:")
        print("  1. 崎岖地形 - 位置 (0, 0)")
        print("  2. 楼梯地形 - 位置 (12, 0)")
        print("  3. 跳跃障碍 - 位置 (0, 12)")
        print("  4. 混合地形 - 位置 (12, 12)")
    
    def run(self):
        """运行演示"""
        print("启动Isaac Lab地形演示...")
        
        # 创建仿真配置
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0/60.0,
            device="cuda",
            gravity=(0.0, 0.0, -9.81),
            render_interval=1,
            enable_scene_query_support=True,
            use_fabric=True,
        )
        
        # 初始化仿真
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        # 设置场景
        self._setup_scene()
        
        # 重置仿真
        self.sim.reset()
        
        print("演示已启动！")
        print("操作说明:")
        print("  - 鼠标左键拖拽: 旋转视角")
        print("  - 鼠标中键拖拽: 平移视角") 
        print("  - 鼠标滚轮: 缩放")
        print("  - 按 ESC 或关闭窗口退出")
        
        # 仿真循环
        frame_count = 0
        try:
            while simulation_app.is_running():
                # 执行仿真步骤
                self.sim.step()
                
                # 每隔一段时间打印信息
                if frame_count % 300 == 0:  # 每5秒
                    print(f"仿真运行中... 帧数: {frame_count}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n接收到中断信号，正在退出...")
        
        print("演示结束")


def main():
    """主函数"""
    print("=" * 60)
    print("Isaac Lab 地形演示")
    print("=" * 60)
    
    # 创建环境配置
    cfg = TerrainDemoEnvCfg()
    
    # 创建并运行环境
    env = TerrainDemoEnv(cfg)
    env.run()
    
    # 关闭仿真应用
    simulation_app.close()


if __name__ == "__main__":
    main()
