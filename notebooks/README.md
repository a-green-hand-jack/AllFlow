# AllFlow 教程和示例

本目录包含AllFlow库的Jupyter notebook教程和示例代码，帮助用户快速上手和深入理解Flow Matching算法。

## 📚 教程内容

### 基础教程
- `01_getting_started.ipynb` - AllFlow快速入门指南
- `02_basic_flow_matching.ipynb` - 基础Flow Matching算法演示
- `03_performance_optimization.ipynb` - 性能优化最佳实践

### 算法演示  
- `flow_matching_demo.ipynb` - 标准Flow Matching演示
- `mean_flow_demo.ipynb` - Mean Flow算法示例
- `conditional_fm_demo.ipynb` - 条件Flow Matching演示
- `rectified_flow_demo.ipynb` - RectifiedFlow迭代优化
- `ot_flow_demo.ipynb` - 最优传输Flow示例

### 高级应用
- `custom_algorithms.ipynb` - 如何实现自定义Flow Matching变体
- `performance_benchmarks.ipynb` - 性能基准测试和对比
- `numerical_analysis.ipynb` - 数值稳定性分析

## 🚀 运行要求

确保已安装以下依赖：
```bash
pip install allflow[examples]
```

包含的依赖：
- jupyter
- matplotlib  
- seaborn
- numpy

## 💡 使用建议

1. **按顺序学习**: 建议从基础教程开始，逐步深入
2. **动手实践**: 修改参数，观察结果变化
3. **性能对比**: 在自己的设备上运行性能测试
4. **扩展实验**: 基于示例代码开发自己的应用

## 🔧 运行notebook

```bash
# 进入项目目录
cd allflow

# 启动Jupyter
jupyter lab notebooks/

# 或使用notebook界面
jupyter notebook notebooks/
```

## 📝 贡献指南

欢迎提交新的教程和示例！请确保：
- 代码清晰易懂，有充分注释
- 包含理论背景说明
- 提供可视化结果
- 遵循项目的代码风格规范 