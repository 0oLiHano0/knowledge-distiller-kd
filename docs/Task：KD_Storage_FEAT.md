**任务ID:** `KD-FEAT-007.1`

---

**一、 背景信息**

项目中同时存在 `FileStorage`（JSON 文件存储）与 `ORMStorage`（SQLite 存储）两套持久化实现。但运行时 `cli.py` 依然默认、唯一地实例化并使用 `FileStorage`，且无动态切换逻辑。本次任务旨在删除或禁用 `FileStorage`，为全项目切换到 `ORMStorage` 做准备。

---

**二、 任务目标 (Objective)**

* 从项目中移除或禁用所有与 `FileStorage` 相关的引用。
* 确保在 CLI 启动时不再实例化 `FileStorage`，同时保留代码目录整洁。

---

**三、 文件位置**

* `knowledge_distiller_kd/cli.py`
* `knowledge_distiller_kd/storage/file_storage.py`
* `knowledge_distiller_kd/storage/__init__.py`
* 可能涉及的测试文件（后续任务清理，这里只注记）

  * `tests/test_file_storage.py`
  * `tests/test_integration.py`（涉及 `FileStorage` 的场景）

---

**四、 子任务列表**

1. **注释或删除 `file_storage.py`**

   * 在 `knowledge_distiller_kd/storage/file_storage.py` 文件顶部添加注释说明“已弃用，保留仅作参考”，或直接删除该文件。
2. **移除 CLI 中的 `FileStorage` 实例化**

   * 打开 `cli.py`，删除以下代码行：

     ```python
     from .storage.file_storage import FileStorage
     …
     storage = FileStorage(base_path=storage_path)
     logger.info(f"FileStorage initialized with base path: {storage_path}")
     ```
3. **清理 `storage/__init__.py`**

   * 如果 `file_storage.py` 被删除，则从 `storage/__init__.py` 中移除对其的导入。
4. **保留但不引用 `FileStorage`**

   * 确保项目中不再有任何 `FileStorage(` 的调用点（可全文搜索并删除）。
5. **运行本地测试**

   * 执行 `pytest -q`，确认所有测试依赖 `FileStorage` 的部分暂时失败（这些测试会在后续 7.4 中重构）。
6. **提交变更**

   * 在 `feature/sqlite-persistence` 分支完成后，提交为一条清晰的 commit 信息：

     ```
     KD-FEAT-007: Disable FileStorage references in preparation for SQLite-only persistence
     ```
   * 推送到远程并更新 PR。

---

**五、 验收标准**

* **代码检查**：项目中不再存在对 `FileStorage` 的任何调用；`file_storage.py` 文件已标记为弃用或删除。
* **CLI 启动**：运行 `python -m knowledge_distiller_kd.cli -h` 不再报找不到 `FileStorage` 的错误。
* **测试反馈**：虽然部分测试会因移除 `FileStorage` 失败，但任务本身的变更不应引入新的、非预期的错误或异常。
* **Git 提交**：变更已提交至 `feature/sqlite-persistence` 分支，并附带符合规范的 commit message。

---
**六、其他**
注意：如在任务执行中发现其他问题，需在任务完成后文字说明
