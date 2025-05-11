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
