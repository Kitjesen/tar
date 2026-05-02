# 地形系统测试文件 (存档)

⚠️ **注意**: 原自定义地形系统已弃用，现在使用Isaac Lab原生地形系统。

本目录包含之前自定义地形系统的测试和示例文件，作为参考保留。

## 📋 文件说明

### 🔍 基础测试

- **`test_terrain_standalone.py`** - 独立地形测试（matplotlib可视化）
  - 测试所有地形生成器
  - 生成地形对比图像
  - 不需要Isaac Lab

### 🎨 可视化测试  

- **`quick_preview.py`** - 快速地形预览（推荐）
  - matplotlib 2D/3D 可视化
  - 交互式地形浏览
  - 不需要Isaac Lab

- **`simple_terrain_viewer.py`** - Isaac Sim 3D查看器
  - 在Isaac Sim中显示地形
  - 需要Isaac Lab环境

- **`test_simple_viewer.py`** - 简化版Isaac查看器
  - 精简版3D查看器
  - 更少依赖，更容易调试

### 📊 高级测试

- **`terrain_visualizer.py`** - 完整地形可视化器
  - 高级3D可视化功能
  - Isaac Lab集成

- **`run_isaac_terrain_demo.py`** - 完整演示脚本
  - 演示所有地形类型
  - Isaac Lab完整功能

### 📖 示例代码

- **`usage_example.py`** - 使用示例
  - 展示API用法
  - 代码示例

### 🔧 环境检查

- **`check_isaac_env.py`** - 环境诊断工具
  - 检查Isaac Lab安装
  - 故障排除

## 🚀 快速开始

### 1. 基础测试（无需Isaac Lab）

```bash
cd tests
python quick_preview.py      # 推荐：快速预览
python test_terrain_standalone.py  # 完整测试
```

### 2. Isaac Lab 3D可视化（需要Isaac Lab环境）

```bash
cd tests
python test_simple_viewer.py     # 简化版
python simple_terrain_viewer.py  # 完整版
```

### 3. 环境检查

```bash
cd tests
python check_isaac_env.py   # 诊断Isaac Lab环境
```

## 📁 推荐测试顺序

1. **基础功能测试**: `python quick_preview.py`
2. **环境检查**: `python check_isaac_env.py`
3. **3D可视化**: `python test_simple_viewer.py`
4. **完整测试**: `python test_terrain_standalone.py`

## 💡 故障排除

- 如果Isaac Lab导入失败，使用 `quick_preview.py`
- 如果3D显示有问题，检查 `check_isaac_env.py` 输出
- 所有matplotlib版本的测试都不需要Isaac Lab
