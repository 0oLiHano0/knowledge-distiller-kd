# 测试说明文档

## 测试目录结构

```
tests/
├── conftest.py          # 测试配置文件
├── test_core_functions.py  # 核心功能测试
├── test_data_generator.py  # 测试数据生成器
├── test_utils.py        # 测试工具
├── run_tests.py         # 测试运行脚本
├── requirements-test.txt # 测试依赖
└── README.md            # 测试说明文档
```

## 测试环境设置

1. 安装测试依赖：

```bash
pip install -r tests/requirements-test.txt
```

2. 确保项目根目录在Python路径中：

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 运行测试

1. 运行所有测试：

```bash
python tests/run_tests.py
```

2. 运行特定测试：

```bash
pytest tests/test_core_functions.py -v
```

3. 运行特定测试用例：

```bash
pytest tests/test_core_functions.py::test_md5_duplicate_detection -v
```

## 测试报告

测试运行完成后，会生成以下报告：

1. 控制台输出：显示测试结果和覆盖率信息
2. HTML报告：在`coverage_report`目录下生成详细的覆盖率报告

## 测试内容

### 核心功能测试

1. MD5重复检测测试
   - 测试MD5重复内容的检测
   - 测试MD5重复内容的处理

2. 语义重复检测测试
   - 测试语义重复内容的检测
   - 测试语义重复内容的处理

3. 文件操作测试
   - 测试文件读取
   - 测试Markdown解析
   - 测试文件保存

4. 决策处理测试
   - 测试决策的保存和加载
   - 测试决策的应用

5. 集成测试
   - 测试完整的工作流程
   - 测试各个组件的协同工作

## 测试数据

测试数据由`test_data_generator.py`生成，包括：

1. MD5重复内容
2. 语义重复内容
3. 正常内容
4. 决策数据

## 注意事项

1. 测试数据会在测试完成后自动清理
2. 测试日志保存在`tests/test.log`文件中
3. 覆盖率报告保存在`coverage_report`目录中
4. 确保有足够的磁盘空间用于生成测试数据和报告 