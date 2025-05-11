## 任务概览
1. 原子任务 6.1：添加 ORM 依赖与基础配置
在 pyproject.toml 或 requirements.txt 中添加 sqlalchemy、alembic
在 knowledge_distiller_kd/core/storage/sqlite_storage.py 中初始化 SQLAlchemy 引擎和会话工厂（SessionLocal）
2. 原子任务 6.2：引入并注册 ORM 模型
将之前定义的 dataclass 风格模型文件（models_sqlalchemy.py）放入 core/models/
在 sqlite_storage.py 中 import 并在 metadata 中注册它们
3. 原子任务 6.3：配置 Alembic 目录与环境
在项目根生成 alembic.ini、alembic/ 目录
编辑 alembic/env.py，让它加载我们 ORM 的 metadata
配置 script_location 指向项目内的迁移脚本目录
4. 原子任务 6.4：生成并验证初版迁移脚本
运行 alembic revision --autogenerate -m "init tables"
检查生成的脚本，确认四张核心表 DDL 与占位表 DDL 都在脚本中
在本地干净环境执行 alembic upgrade head，并用 SQLite 客户端验证表已创建
5. 原子任务 6.5：在引擎中集成持久化调用
在 KnowledgeDistillerEngine.run_analysis() 尾部，把内存中的分析结果写入数据库
在 engine.save_results() 中调用 Session 的事务，批量插入 Document、Block、Analysis、Decision
6. 原子任务 6.6：编写持久化单元与集成测试
在内存数据库 (sqlite:///:memory:) 上测试：
CRUD：创建 Document，关联 Block、Analysis、Decision 并查询
事务回滚：故意在中间抛错，检查数据回滚情况
在临时目录下生成真实 SQLite 文件，验证 alembic upgrade head 与 ORM 操作一致


### 原子任务 6.1：添加 ORM 依赖与基础配置

**文件位置**

* 项目根目录：`pyproject.toml` 或 `requirements.txt`
* `knowledge_distiller_kd/core/storage/sqlite_storage.py`

**目标**

* 在项目依赖文件中添加 `sqlalchemy` 和 `alembic`（开发依赖）
* 在 `sqlite_storage.py` 中：

  1. 导入 SQLAlchemy：

     ```python
     from sqlalchemy import create_engine
     from sqlalchemy.orm import sessionmaker
     ```
  2. 初始化数据库引擎：

     ```python
     engine = create_engine(
         config.DATABASE_URL,  # e.g. 'sqlite:///./data/kd_tool.db'
         connect_args={"check_same_thread": False},
         echo=False  # 可根据调试需求打开
     )
     ```
  3. 创建会话工厂：

     ```python
     SessionLocal = sessionmaker(
         autoflush=False,
         autocommit=False,
         bind=engine
     )
     ```
  4. 在模块顶部或专门的 `init_db()` 函数中，确保目录存在并可调用：

     ```python
     def init_db():
         from core.models import Base  # SQLAlchemy ORM 基类
         Base.metadata.create_all(bind=engine)
     ```

**输入示例**

* `pyproject.toml` 增加：

  ```toml
  [tool.poetry.dev-dependencies]
  sqlalchemy = "^2.0"
  alembic = "^1.10"
  ```

**验收标准**

1. `poetry install` 或 `pip install -r requirements.txt` 能安装 `sqlalchemy`、`alembic` 而无报错
2. 在 Python 交互式环境中：

   ```python
   >>> from knowledge_distiller_kd.core.storage.sqlite_storage import init_db, engine, SessionLocal
   >>> init_db()
   >>> session = SessionLocal()
   >>> session  # 应创建成功，无异常
   Session<...>
   ```
3. 在本地运行 `init_db()` 后，指定数据目录内生成 SQLite 数据文件（若使用文件 URL）或在内存模式下无错误
4. 编写并通过单元测试：

   * 验证 `SessionLocal` 可以创建会话并开始事务
   * 验证 `init_db()` 在干净环境下创建表结构（可在内存模式下断言 `engine.table_names()` 包含核心表名）

### 原子任务 6.2：引入并注册 ORM 模型（细化）

**文件位置**

* ORM 模型文件：`knowledge_distiller_kd/storage/models_sqlalchemy.py`
* 存储模块：`knowledge_distiller_kd/storage/sqlite_storage.py`

**子任务列表**

1. **创建 `models_sqlalchemy.py` 文件**

   * 在 `knowledge_distiller_kd/storage/` 目录下新建 `models_sqlalchemy.py`
   * 导入必要模块：

     ```python
     from sqlalchemy.orm import declarative_base, relationship
     from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, func
     ```
   * 定义基础：

     ```python
     Base = declarative_base()
     ```

2. **定义 `Document` 模型**

   * 在 `models_sqlalchemy.py` 中：

     ```python
     class Document(Base):
         __tablename__ = 'documents'
         id = Column(Integer, primary_key=True, autoincrement=True)
         path = Column(String, unique=True, nullable=False)
         file_hash = Column(String, nullable=False)
         type = Column(String)
         size = Column(Integer)
         ctime = Column(DateTime)
         mtime = Column(DateTime)
         ingest_time = Column(DateTime, server_default=func.now())
         status = Column(String)
         blocks = relationship('Block', back_populates='document', cascade='all, delete-orphan')
     ```

3. **定义 `Block` 模型**

   * 在 `models_sqlalchemy.py` 中紧接上部：

     ```python
     class Block(Base):
         __tablename__ = 'blocks'
         id = Column(Integer, primary_key=True, autoincrement=True)
         document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
         content_hash = Column(String)
         simhash = Column(String)
         text = Column(Text)
         raw_element_type = Column(String)
         processing_status = Column(String)
         metadata = Column(JSON)
         document = relationship('Document', back_populates='blocks')
         analyses = relationship('Analysis', back_populates='block', cascade='all, delete-orphan')
     ```

4. **定义 `Analysis` 模型**

   * 在 `models_sqlalchemy.py` 中继续添加：

     ```python
     class Analysis(Base):
         __tablename__ = 'analyses'
         id = Column(Integer, primary_key=True, autoincrement=True)
         block_id = Column(Integer, ForeignKey('blocks.id', ondelete='CASCADE'), nullable=False)
         analysis_type = Column(String, nullable=False)
         score = Column(JSON)
         details = Column(JSON)
         block = relationship('Block', back_populates='analyses')
     ```

5. **定义 `Decision` 模型**

   * 在 `models_sqlalchemy.py` 中最后添加：

     ```python
     class Decision(Base):
         __tablename__ = 'decisions'
         id = Column(Integer, primary_key=True, autoincrement=True)
         block_id = Column(Integer, ForeignKey('blocks.id', ondelete='CASCADE'), nullable=False)
         decision_type = Column(String, nullable=False)
         duplicate_of_block_id = Column(Integer, ForeignKey('blocks.id'))
         timestamp = Column(DateTime, server_default=func.now())
         comment = Column(Text)
         block = relationship('Block', foreign_keys=[block_id])
     ```

6. **更新 `sqlite_storage.py`**

   * 在文件顶部导入模型和基础类：

     ```python
     from .models_sqlalchemy import Base, Document, Block, Analysis, Decision
     ```
   * 确保 `init_db()` 使用了新导入的 `Base.metadata.create_all(bind=engine)`。

**验收标准**

* `models_sqlalchemy.py` 文件中依次定义了 `Base`、`Document`、`Block`、`Analysis`、`Decision` 6 个实体。
* `sqlite_storage.py` 成功导入上述模型，并 `init_db()` 能创建对应表。
* 单元测试 `tests/storage/test_models_registration.py` 通过验证 CRUD 操作：

  ```python
  session = SessionLocal()
  doc = Document(path='test.md', file_hash='abc')
  session.add(doc)
  session.commit()
  assert session.query(Document).count() == 1
  ```
* CI 中所有新增测试绿色通过。

### 原子任务 6.3：配置 Alembic 环境

**文件位置**

* 项目根目录（新增）: `alembic.ini`
* 项目根目录（新增）: `alembic/`
* Alembic 环境脚本：`alembic/env.py`

**目标**

1. 在项目根运行 `alembic init alembic`，生成 `alembic.ini` 和 `alembic/` 目录结构。
2. 修改 `alembic.ini`：

   * 设置 `sqlalchemy.url` 为项目的 `DATABASE_URL`（可使用 env var 或写入 `core/constants.py` 中的值）。
3. 编辑 `alembic/env.py`：

   * 导入项目的 ORM metadata：

     ```python
     import sys
     from os.path import dirname, abspath
     sys.path.append(dirname(dirname(abspath(__file__))))

     from knowledge_distiller_kd.storage.models_sqlalchemy import Base
     target_metadata = Base.metadata
     ```
   * 在 `run_migrations_online()` 方法中，使用 `config.get_main_option("sqlalchemy.url")` 获取 `DATABASE_URL`，并创建 `engine` 进行在线迁移。
4. 在 `alembic/` 根目录下新建空的 `versions/` 文件夹，用于后续存放迁移脚本。

**实现细节**

* `alembic.ini` 中的路径相对性要准确，若使用 env var，可写：

  ```ini
  sqlalchemy.url = sqlite:///./data/kd_tool.db
  ```
* 确保 `env.py` 中的 `target_metadata` 指向正确的 `Base.metadata`，否则无法生成自动迁移脚本。
* 保持 `versions/` 目录可写权限。

**验收标准**

1. 执行 `alembic current` 时，不报错并显示当前版本（可能为空）。
2. `alembic revision --autogenerate -m "init tables"` 能生成初版迁移脚本到 `alembic/versions/`。
3. 运行 `alembic upgrade head` 后，数据库中创建了 `documents`, `blocks`, `analyses`, `decisions` 四张表（可使用 SQLite CLI 或 `engine.table_names()`).
4. `git diff` 中包含 `alembic.ini`、`alembic/env.py` 与空的 `alembic/versions/` 目录结构。
5. CI 流水线上模拟运行 `alembic upgrade head` 不报错。


### 原子任务 6.4：自动生成并验证初版迁移脚本

**文件位置**

* Alembic 目录：`alembic/versions/`
* 测试目录：`tests/storage/`

**目标**

1. 使用 Alembic 的自动生成功能，创建初始化迁移脚本（包含核心表的创建）
2. 验证生成的脚本与 ORM 模型一致
3. 在干净环境下执行迁移，并确认数据库结构正确

**子任务列表**

1. **生成迁移脚本**

   * 在项目根执行：

     ```bash
     alembic revision --autogenerate -m "init tables"
     ```
   * 确认在 `alembic/versions/` 目录下生成了新的 `.py` 文件

2. **审查迁移脚本内容**

   * 打开新生成的脚本，检查 `upgrade()` 函数中包含 `op.create_table('documents', ...)`、`op.create_table('blocks', ...)`、`op.create_table('analyses', ...)`、`op.create_table('decisions', ...)` 四个表的创建语句
   * 在 `downgrade()` 中对应包含 `op.drop_table(...)` 调用

3. **编写自动化测试**

   * 在 `tests/storage/test_migration_autogenerate.py` 中，编写测试用例：

     ```python
     def test_autogenerate_migration_creates_tables(tmp_path, monkeypatch):
         # 设置 ALEMBIC_CONFIG 指向项目 alembic.ini
         # 调用 alembic.command.revision(autogenerate=True)
         # 断言 alembic/versions 目录中文件只增多一份
         # 读取新脚本内容并断言包含核心表名
     ```
   * 确保该测试在CI环境中运行，并能通过

4. **执行迁移测试**

   * 在测试或脚本中，执行 `alembic upgrade head`
   * 使用 SQLAlchemy `engine.table_names()`（或 `inspect(engine).get_table_names()`）断言表列表包含

     ```python
     ['documents', 'blocks', 'analyses', 'decisions']
     ```

**验收标准**

* 生成的迁移脚本准确包含四张核心表的 CREATE 语句
* 自动化测试 `tests/storage/test_migration_autogenerate.py` 在干净环境下通过
* 执行 `alembic upgrade head` 能在新数据库中创建核心表
* CI 流水线中相关测试与迁移步骤均绿灯


### 原子任务 6.5：在引擎中集成持久化调用

**背景：已完成任务回顾**

* **6.1**：添加了 SQLAlchemy 与 Alembic 依赖，完成了 `sqlite_storage.py` 基础配置和 `init_db()` 初始化功能。
* **6.2**：定义并注册了 `Document`、`Block`、`Analysis`、`Decision` 四个 ORM 模型，确保 `init_db()` 能创建对应表。
* **6.3**：配置了 Alembic 环境，生成并验证了初版迁移脚本，确保数据库模式与 ORM 模型同步。
* **6.4**：使用 Alembic 自动生成功能，优化并验证了迁移脚本，增加了索引，编写了测试和工具脚本以重置/验证数据库。

**目标**

* 将 `run_analysis()` 的内存分析结果持久化到 SQLite 数据库中：

  1. 在分析开始前调用 `init_db()`（若未初始化）
  2. 批量插入新扫描的 `Document` 记录
  3. 插入每个 `Block` 及其 `Analysis` 结果
* 在 `save_results()` 中统一提交事务，并在异常时回滚

**子任务列表**

1. **更新 `KnowledgeDistillerEngine`**

   * 在 `run_analysis()` 开头调用 `init_db()`
   * 在流程末尾，将内存 `documents`、`blocks`、`analyses` 汇总为 ORM 实例，并通过 `SessionLocal()` 批量 `session.add_all()`
2. **实现 `save_results()`**

   * 接收 `analysis_results: Dict` 与 `decisions: List`，转换为 ORM 对象
   * 在一个事务中 `session.begin()` 执行所有插入/更新操作，调用 `session.commit()`
   * 在 `except` 块中调用 `session.rollback()` 并重新抛出异常
3. **编写单元测试**

   * 在内存数据库（`sqlite:///:memory:`）上测试：

     ```python
     # 模拟 run_analysis() 返回内存结果
     engine = KnowledgeDistillerEngine(input_dir=str(tmp_path), skip_prefilter=True)
     engine.run_analysis()
     # 断言 session.query(Document).count() == expected
     ```
   * 测试事务回滚：在中途人为抛出异常，验证数据库无部分写入

**验收标准**

* `run_analysis()` 执行后，通过 ORM 查询能在数据库中找到对应记录
* `save_results()` 在正常和异常场景均能正确提交或回滚
* 新增测试 `tests/storage/test_persistence.py` 全部通过
* CI 流水线绿色，无回归错误


### 原子任务 6.6：对齐表名与测试预期

**问题现象**
现有的 SQLite 核心表在迁移脚本和 ORM 模型中定义为：

* `documents`
* `blocks`
* `analyses`
* `decisions`

但单元测试（`tests/test_sqlite_storage.py`）检查的是：

```python
required_tables = ["files", "blocks", "analysis_results", "user_decisions"]
```

导致初始化测试 `test_init_db` 断言失败，提示表 `files`, `analysis_results`, `user_decisions` 未被创建。

**目标**

* 将核心表名修改为与测试一致：

  * `documents`   → `files`
  * `analyses`    → `analysis_results`
  * `decisions`   → `user_decisions`
  * 保留 `blocks` 不变
* 同步更新 ORM 模型、Alembic 迁移脚本以及任何引用表名的代码。

**子任务列表**

1. **更新 ORM 模型**

   * 在 `storage/models_sqlalchemy.py`（或相应模型文件）中修改：

     ```python
     class Document(Base):
         __tablename__ = 'files'
         # ...

     class Analysis(Base):
         __tablename__ = 'analysis_results'
         # ...

     class Decision(Base):
         __tablename__ = 'user_decisions'
         # ...
     ```
2. **更新 Alembic 初始迁移脚本**

   * 在 `alembic/versions/<initial>_initial_tables.py` 中，将 `documents`, `analyses`, `decisions` 的 `op.create_table()` 调整为新表名。
3. **添加兼容性迁移（如需）**

   * 如果已有线上数据，可编写新的迁移脚本将旧表重命名为新表。若仅为开发环境，直接修改初始脚本即可。
4. **更新存储层引用**

   * 在 `sqlite_storage.py`、查询/inspect 相关代码中，保证对新表名的访问一致。
5. **修改测试用例（如必要）**

   * 确保 `test_init_db` 中的 `required_tables` 与新表名一致；如有硬编码字段名，也一并更新。

**验收标准**

* `init_db()` 创建的表名为：

  ```text
  files, blocks, analysis_results, user_decisions
  ```
* `pytest tests/test_sqlite_storage.py` 全部通过（特别是 `test_init_db`）。
* CI 流水线绿色，无与表名不一致相关的失败。


### 原子任务 6.7：修复 CzkawkaAdapter 命令构建与 JSON 解析逻辑

**背景**
当前 `CzkawkaAdapter` 的 `_build_command()` 与 `_parse_czkawka_json_to_dtos()` 方法存在以下问题：

1. **命令构建过于冗余**：

   * `_build_command()` 在默认或自定义参数时，均会附加额外 flag (`--directories`, `-p`, `-m`)，导致测试中预期的列表比实际结果多。
2. **JSON 解析失败**：

   * `_parse_czkawka_json_to_dtos()` 直接对 `data` 使用 `.items()`，对列表结构无法解析，所有解析测试均返回空结果。

**目标**

* 精简 `_build_command()`：仅包含 `czkawka_cli_path`、`czkawka_args`（默认为空），和目标目录路径。
* 强化 JSON 解析：正确处理顶层列表结构，使用 `.get('files', [])` 遍历每个对象并提取路径/大小数据，过滤空 `files` 组并保留单文件组（或依据测试要求保留）。

**子任务列表**

1. **调整 `_build_command()` 实现**

   * 默认 `self.config.get('czkawka_args', [])` ；
   * 返回 `[self.czkawka_cli_path, *args, str(dir_path)]`。
   * 移除所有其它硬编码 flag（`--directories`, `--json`, `-p`, `-m`）；
2. **重写 `_parse_czkawka_json_to_dtos()`**

   ```python
   def _parse_czkawka_json_to_dtos(self, data: List[Dict]) -> List[DuplicateFileGroupDTO]:
       groups = []
       for entry in data:
           files = entry.get('files', [])
           if not files:
               continue
           dto = DuplicateFileGroupDTO(
               file_paths=[f['path'] for f in files],
               sizes=[f['size'] for f in files]
           )
           groups.append(dto)
       return groups
   ```
3. **更新 `filter_unique_files()` 和 `scan_directory_for_duplicates()`**

   * 确保新命令生成方式与测试一致；
   * 确保 JSON 解析返回正确 DTO 列表；
4. **修改配置默认值**

   * 在 `czkawka_adapter.py` 顶部，将默认 `czkawka_args` 设置为空列表；
5. **编写/更新测试**

   * 确认 `TestCzkawkaAdapter` 中所有构建命令测试通过；
   * 确认 JSON 解析相关测试用例 (`test_parse_valid_czkawka_json_to_dtos`, `test_parse_malformed_json...` 等) 均通过；
6. \*\*执行所有 `tests/prefilter/test_czkawka_adapter.py` 中的测试，并确保存量测试全部通过。

**验收标准**

* `_build_command()` 仅包含路径参数，与测试期望 `expected = [cli_path, *args, dir]` 完全一致；
* JSON 解析方法能正确处理多种结构，所有相关测试通过；
* `scan_directory_for_duplicates()` 能返回符合测试案例的 DTO 列表长度；
* CI 流水线中 `tests/prefilter/test_czkawka_adapter.py` 无失败项。


### 原子任务 6.8：修复持久化方法中的字段映射

**问题现象**
在持久化分析结果的 `save_results()` 方法中，`Block` ORM 实例化依然使用了旧的关键字 `document_id`（以及 `raw_element_type`），而模型已更新为 `file_id` 和 `block_type`。执行流程时会抛出：

```
TypeError: 'document_id' is an invalid keyword argument for Block
```

**目标**
将 `save_results()`（及相关批量插入逻辑）中的关键字参数更名为与 ORM 模型一致，确保持久化时不再抛错，并正确写入数据库。

**子任务列表**

1. **更新 Block 实例化**

   * 在 `knowledge_distiller_kd/core/engine.py` 的 `save_results()` 中，替换所有构造 `Block(...)` 时的参数：

     * `document_id=` → `file_id=`
     * `raw_element_type=` → `block_type=`
2. **检查并更新 Analysis 和 Decision**

   * 确认 `Analysis` 和 `Decision` 实例的构造参数与 ORM 模型字段一致，修正 `block_id`、`duplicate_of_block_id` 等关键字。
3. **移除多余参数**

   * 全局搜索确认不再使用 `document_id`，移除残留的旧参数。
4. **编写/更新测试**

   * 在 `tests/storage/test_persistence.py` 中，验证：

     * 执行 `save_results()` 后，`blocks` 表的 `file_id`、`block_type` 列均能正确保存值。
     * 不再抛出关键字参数无效的异常。
5. **验证与回归**

   * 运行 `pytest tests/storage/test_persistence.py`，确保该模块测试全绿。
   * 全量执行 `pytest`，确认无新回归。

**验收标准**

* 执行分析并保存结果（CLI 或 API）时，不再出现 `invalid keyword argument for Block` 错误。
* `blocks` 表中的 `file_id`、`block_type`、`text` 等字段能正确写入数据。
* 持久化相关测试（尤其是 `test_save_results_transaction`、`test_save_results_rollback`）全部通过。
* CI 流水线绿色，无与字段映射相关的失败。
