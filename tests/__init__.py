"""AllFlow测试套件

AllFlow项目的完整测试套件，包含单元测试、集成测试、性能基准测试
和数值验证测试。

测试组织：
- unit/: 单元测试，测试各个模块的独立功能
- integration/: 集成测试，测试模块间的协作
- benchmark/: 性能基准测试，评估计算效率
- fixtures/: 测试数据和工具函数

测试覆盖：
- 算法正确性：数学公式实现的正确性
- 数值稳定性：边界条件和异常输入的处理
- 性能效率：批量处理和GPU加速的效果
- 接口一致性：不同算法间的API兼容性

运行方式：
    pytest                              # 运行所有测试
    pytest --cov=src/allflow           # 运行测试并生成覆盖率
    pytest -m benchmark                # 只运行性能测试
    pytest -m "not slow"               # 跳过慢速测试

Author: AllFlow Contributors
"""

# 测试工具和配置将在测试实现时添加 