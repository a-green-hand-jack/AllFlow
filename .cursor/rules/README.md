# AllFlow 项目 Cursor Rules 说明

## 规则文件分类与应用策略

### 🔄 Always Apply (总是应用的规则)

这些规则在所有代码编写过程中都会自动应用：

#### 1. `project-goals.mdc` 
- **核心目标定位**: 确保所有开发都围绕Flow Matching算法核心
- **专注度原则**: 避免偏离项目本质（不做神经网络、不做数据处理）
- **性能和科学原则**: 始终强调计算效率和数学正确性

#### 2. `code-style-standards.mdc`
- **代码质量标准**: 类型注解、日志系统、错误处理等基础要求
- **模块结构规范**: src布局、分层设计等架构原则
- **编程规范**: PEP 8、Google docstrings等代码风格

#### 3. `tensor-optimization.mdc`
- **性能核心要求**: 禁止Python循环、强制张量操作
- **优化策略**: 内存效率、计算效率、GPU优化
- **代码审查检查点**: 每次开发都需要验证的性能指标

#### 4. `device-compatibility.mdc`
- **跨平台兼容性**: M4 Mac开发环境与Linux部署环境
- **设备自动检测**: 智能选择CUDA、MPS或CPU后端
- **性能优化**: 针对不同设备的特定优化策略

### 📋 Conditional Apply (按需应用的规则)

这些规则在特定开发阶段或特定文件类型时应用：

#### 5. `algorithm-implementation.mdc`
- **应用时机**: 实现具体Flow Matching算法时
- **内容**: 统一接口、数学正确性、数值稳定性要求
- **验证标准**: 算法特定的测试和验证要求

#### 6. `testing-coverage.mdc`
- **应用时机**: 编写测试代码或审查测试覆盖率时
- **内容**: 测试框架、覆盖率要求、测试类型和组织
- **工具配置**: pytest配置、CI/CD集成

#### 7. `documentation-standards.mdc`
- **应用时机**: 编写文档、README或代码注释时
- **内容**: Docstring规范、数学公式表示、项目级文档
- **文档类型**: API文档、教程、版本日志

#### 8. `ref-repo.mdc`
- **应用时机**: 需要参考项目结构或查阅理论文档时
- **内容**: ProtRepr项目参考、Flow Matching理论文档
- **技术栈指引**: 推荐工具和不参考的方面

## 设计原理说明

### 为什么某些规则总是应用？

**Always Apply规则**代表项目的**核心DNA**：
- **项目目标**: 永远不能偏离Flow Matching算法核心
- **代码质量**: 基础编程规范必须始终遵循  
- **性能要求**: 张量优化是项目的生命线，任何代码都不能违反

### 为什么某些规则按需应用？

**Conditional Apply规则**代表**专业领域知识**：
- **算法实现**: 只在实现具体算法时需要详细的数学规范
- **测试标准**: 只在编写测试时需要具体的覆盖率和框架要求
- **文档规范**: 只在编写文档时需要详细的格式和内容要求
- **参考资源**: 只在需要查阅或对比时使用

## 使用指南

### 开发新算法时
1. Always Apply规则自动生效
2. 主动参考 `algorithm-implementation.mdc`
3. 同时考虑 `testing-coverage.mdc` 编写对应测试

### 优化性能时  
1. `tensor-optimization.mdc` 的所有要求都会自动检查
2. 参考 `testing-coverage.mdc` 中的性能测试要求

### 编写文档时
1. 遵循 `documentation-standards.mdc` 的详细规范
2. Always Apply规则确保代码质量不降低

### 项目结构调整时
1. 参考 `ref-repo.mdc` 中的ProtRepr项目结构
2. 确保符合 `code-style-standards.mdc` 的模块要求

## 规则维护

这套规则体系应该：
- **稳定性**: Always Apply规则应该很少修改
- **灵活性**: Conditional Apply规则可以根据项目发展调整
- **一致性**: 所有规则之间不应该有冲突
- **完整性**: 覆盖AllFlow项目开发的所有关键方面

---

**下一步**: 开始创建项目基础架构，从 `pyproject.toml` 和目录结构开始！ 