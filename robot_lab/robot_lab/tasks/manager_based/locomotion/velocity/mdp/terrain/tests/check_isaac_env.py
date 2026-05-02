#!/usr/bin/env python3

"""
Isaac Lab 环境检查工具
检查Isaac Lab是否正确安装和配置
"""

import sys
import os

def check_python_environment():
    """检查Python环境"""
    print("🐍 Python环境信息:")
    print(f"   Python版本: {sys.version}")
    print(f"   Python路径: {sys.executable}")
    print(f"   虚拟环境: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")
    
    # 检查PYTHONPATH
    python_path = os.environ.get('PYTHONPATH', '')
    if python_path:
        print(f"   PYTHONPATH: {python_path}")
    else:
        print("   PYTHONPATH: 未设置")

def check_isaac_imports():
    """检查Isaac相关模块导入"""
    print("\n🔍 Isaac模块检查:")
    
    # 检查不同的导入方式
    import_tests = [
        ("isaaclab", "from isaaclab.app import AppLauncher"),
        ("isaaclab.sim", "import isaaclab.sim as sim_utils"),
        ("omni.isaac.lab", "from omni.isaac.lab.app import AppLauncher"),
        ("omni.isaac.lab.sim", "import omni.isaac.lab.sim as sim_utils"),
        ("omni.isaac.core", "import omni.isaac.core"),
        ("omni", "import omni"),
    ]
    
    for module_name, import_code in import_tests:
        try:
            exec(import_code)
            print(f"   ✅ {module_name}: 可用")
        except ImportError as e:
            print(f"   ❌ {module_name}: 不可用 ({e})")
        except Exception as e:
            print(f"   ⚠️  {module_name}: 错误 ({e})")

def check_isaac_paths():
    """检查Isaac相关路径"""
    print("\n📁 Isaac路径检查:")
    
    # 检查常见的Isaac安装路径
    common_paths = [
        "/isaac-sim",
        "~/.local/share/ov/pkg/isaac_sim-*",
        "/opt/nvidia/isaac_sim",
        "~/isaac-sim",
        "/home/bsrl/isaac-sim",
    ]
    
    for path in common_paths:
        expanded_path = os.path.expanduser(path)
        if "*" in expanded_path:
            import glob
            matches = glob.glob(expanded_path)
            if matches:
                print(f"   ✅ 找到Isaac Sim: {matches[0]}")
            else:
                print(f"   ❌ 未找到: {path}")
        else:
            if os.path.exists(expanded_path):
                print(f"   ✅ 找到Isaac Sim: {expanded_path}")
            else:
                print(f"   ❌ 未找到: {path}")

def check_conda_packages():
    """检查conda包"""
    print("\n📦 已安装包检查:")
    
    import subprocess
    try:
        # 检查isaac相关包
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        packages = result.stdout
        
        isaac_packages = [line for line in packages.split('\n') if 'isaac' in line.lower()]
        if isaac_packages:
            print("   Isaac相关包:")
            for pkg in isaac_packages:
                print(f"     {pkg}")
        else:
            print("   ❌ 未找到Isaac相关包")
            
        # 检查omni相关包
        omni_packages = [line for line in packages.split('\n') if 'omni' in line.lower()]
        if omni_packages:
            print("   Omni相关包:")
            for pkg in omni_packages:
                print(f"     {pkg}")
        else:
            print("   ❌ 未找到Omni相关包")
            
    except Exception as e:
        print(f"   ❌ 无法检查包列表: {e}")

def check_project_structure():
    """检查项目结构"""
    print("\n📂 项目结构检查:")
    
    current_dir = os.getcwd()
    print(f"   当前目录: {current_dir}")
    
    # 检查关键文件
    key_files = [
        "rough_terrain.py",
        "stairs_terrain.py", 
        "gap_terrain.py",
        "terrain_manager.py",
        "quick_preview.py"
    ]
    
    for file in key_files:
        if os.path.exists(file):
            print(f"   ✅ {file}: 存在")
        else:
            print(f"   ❌ {file}: 缺失")

def provide_solutions():
    """提供解决方案"""
    print("\n💡 解决方案建议:")
    
    print("\n1. 如果Isaac Lab未安装:")
    print("   # 方法1: 从源码安装Isaac Lab")
    print("   git clone https://github.com/isaac-sim/IsaacLab.git")
    print("   cd IsaacLab")
    print("   ./install.sh")
    
    print("\n2. 如果Isaac Sim未安装:")
    print("   # 下载并安装Isaac Sim 2023.1+")
    print("   # https://developer.nvidia.com/isaac-sim")
    
    print("\n3. 设置环境变量:")
    print("   export ISAAC_SIM_PATH=/path/to/isaac-sim")
    print("   export PYTHONPATH=$ISAAC_SIM_PATH:$PYTHONPATH")
    
    print("\n4. 激活正确的conda环境:")
    print("   conda activate thunder2")
    
    print("\n5. 临时解决方案 - 使用matplotlib版本:")
    print("   python quick_preview.py")

def main():
    """主函数"""
    print("🔧" + "=" * 50 + "🔧")
    print("    Isaac Lab 环境诊断工具")
    print("🔧" + "=" * 50 + "🔧")
    
    check_python_environment()
    check_isaac_imports()
    check_isaac_paths()
    check_conda_packages()
    check_project_structure()
    provide_solutions()
    
    print("\n" + "=" * 60)
    print("诊断完成!")

if __name__ == "__main__":
    main()
