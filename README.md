# Knowledge Distiller (KD)
知识蒸馏工具 - 一个用于检测和处理文档重复内容的智能工具。

## 项目概述
Knowledge Distiller (KD) 是一个强大的文档内容分析工具，采用分层架构设计(UI, Core, Analysis, Processing, Storage)，使用先进的语义分析技术，能够：
- 通过Czkawka进行文件级预过滤（待嵌入流程）
- 实现MD5 → SimHash（开发中） → SBERT三层内容去重
- 支持SQLite作为核心数据存储（开发中）
- 提供灵活的命令行交互界面
- 提供UI界面(开发中)

## 使用说明

### CLI参数

- `scan`: 扫描指定目录查找重复文件 (参数: 目录路径)
- `dedup`: 执行去重处理 (参数: 输入目录)
- `apply`: 应用决策结果 (参数: 决策文件路径)
- `--similarity-threshold`: 语义相似度阈值 (默认: 0.8, 范围: 0.1-1.0)
- `--model`: 使用的语义模型 (默认: `paraphrase-multilingual-MiniLM-L12-v2`)
- `--input-dir`: 输入目录路径 (默认: `input/`)
- `--output-dir`: 输出目录路径 (默认: `output/`) 
- `--skip-semantic`: 跳过语义分析阶段 (布尔标志)
- `--timeout`: Czkawka扫描超时时间(秒) (默认: 60)
- `--min-size`: 最小处理文件大小(KB) (默认: 1)
- `--db-path`: SQLite数据库文件路径 (默认: `data/kd.db`)
- `--verbose`: 显示详细日志 (布尔标志)



### 开发进度与路线图 (2025-06-15更新)

**当前迭代 (Iter.III - SQLite集成):**
- [x] **CzkawkaAdapter**: 完成文件级预过滤和测试 (I.A-I.E)
- [x] **BlockMerger**: 优化代码块合并算法 (Iter.0)
- [ ] **SqliteStorage**: 实现核心表结构和数据访问 (III.A-III.C)
  - [ ] 文档表(`documents`)和重复文件表(`duplicate_files`)创建
  - [ ] Czkawka扫描结果存储接口
  - [ ] 文件路径检索功能

**近期计划:**
- [ ] **SimHashAnalyzer**: 实现内容指纹计算 (Iter.II)
- [ ] **Pipeline集成**: 连接Czkawka预过滤和内容分析 (Iter.IV)
- [ ] **CLI增强**: 完善`scan-files`和`dedup`命令 (Iter.VI)

**已完成里程碑:**
- [x] **基础架构**: 项目初始化与核心模块划分
- [x] **测试覆盖**: 关键模块单元测试覆盖率>90%
- [x] **CI/CD**: 实现自动化测试和代码质量检查

**长期技术规划:**
- [ ] **性能基准**: 建立1GB测试集和性能监控 (Phase P)
- [ ] **插件架构**: 为OCR和图片处理预留接口 (Phase E)
- [ ] **企业功能**: 批量处理和大规模文档治理支持

