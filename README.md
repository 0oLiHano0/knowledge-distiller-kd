# Knowledge Distiller (KD)

知识蒸馏工具 - 一个用于检测和处理文档重复内容的智能工具。

## 项目概述

Knowledge Distiller (KD) 是一个强大的文档内容分析工具，专门用于检测和处理文档中的重复内容。它使用先进的语义分析技术，不仅能够识别完全相同的内容，还能发现语义相似的内容片段。

### 主要特性

- 精确重复检测：使用 MD5 哈希算法检测完全相同的内容
- 语义相似度分析：使用 Sentence Transformers 进行语义相似度计算
- 交互式决策：提供友好的命令行界面进行内容处理
- 批量处理：支持批量处理多个文档
- 缓存机制：使用向量缓存提高性能
- 多语言支持：支持中文和英文文档

## 安装指南

### 系统要求

- Python 3.8+
- pip 包管理器

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/knowledge-distiller-kd.git
cd knowledge-distiller-kd
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装开发依赖（可选）：
```bash
pip install -r requirements-dev.txt
```

## 使用方法

### 基本用法

1. 准备输入文件：
   - 将需要处理的 Markdown 文件放入 `input` 目录

2. 运行工具：
```bash
python -m knowledge_distiller_kd.core.kd_tool_CLI input_dir
```

3. 交互式处理：
   - 工具会显示找到的重复内容
   - 使用命令进行处理：
     - `k1d2`: 保留第一个，删除第二个
     - `k2d1`: 保留第二个，删除第一个
     - `k`: 保留两个块
     - `d`: 删除两个块
     - `r`: 重置决策
     - `save`: 保存当前决策
     - `q`: 退出

### 高级配置

可以通过修改以下参数自定义工具行为：

- `similarity_threshold`: 语义相似度阈值（默认：0.85）
- `model_name`: 使用的语义模型（默认：paraphrase-multilingual-MiniLM-L12-v2）
- `batch_size`: 批处理大小（默认：32）

## 开发指南

### 项目结构

```
knowledge_distiller_kd/
├── core/
│   ├── __init__.py
│   ├── constants.py      # 常量定义
│   ├── semantic_analyzer.py  # 语义分析器
│   ├── md5_analyzer.py   # MD5分析器
│   ├── kd_tool_CLI.py    # 命令行接口
│   ├── error_handler.py  # 错误处理
│   └── utils.py         # 工具函数
├── tests/
│   ├── __init__.py
│   ├── test_constants.py
│   ├── test_semantic_analyzer.py
│   └── ...
└── docs/
    └── ...
```

### 测试

运行测试：
```bash
python -m pytest tests/
```

运行特定测试：
```bash
python -m pytest tests/test_semantic_analyzer.py -v
```

### 开发进度

- [x] 基础框架搭建
- [x] MD5 精确重复检测
- [x] 语义分析器实现
- [x] 测试用例编写
- [x] 文档完善
- [ ] 性能优化
- [ ] Web 界面开发
- [ ] API 接口开发

### 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 常见问题

1. 语义分析功能不可用？
   - 确保已安装 sentence-transformers：`pip install sentence-transformers`
   - 检查模型下载是否完成

2. 向量计算很慢？
   - 考虑减小 batch_size
   - 确保使用了向量缓存
   - 如果可能，使用 GPU 加速

## 更新日志

### v1.0.0 (2024-05-01)
- 初始版本发布
- 实现基本功能：MD5分析和语义分析
- 添加测试用例和文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目维护者：[Your Name]
- 邮箱：[your.email@example.com]
- 项目主页：[GitHub Repository URL] 