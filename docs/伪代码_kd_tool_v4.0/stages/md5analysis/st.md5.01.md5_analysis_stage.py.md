```python
# ------------------------------------------------------------------------------
# 文件名: kd_tool/stages/md5analysis/md5_analysis_stage.py
# 模块: MD5 分析阶段 (MD5AnalysisStage)
# 描述:
#   此模块负责计算内容块的 MD5 哈希值，并在同一阶段内找出 MD5 完全相同的块，
#   生成相应的成对分析结果 (AnalysisResultDTO)。
#   它实现了计算与比较的双重职责，遵循我们最终的架构决策。
# 架构约束:
#   - 必须实现 StageInterface。
#   - 必须通过构造函数接收 Logger, StorageInterface, MD5AnalysisStageSettings。
#   - 必须使用 StorageInterface 进行所有 *外部* 数据持久化操作 (但此阶段主要操作 Context)。
#   - 必须抛出 MD5InputError 或 MD5CalculationError。
#   - 必须是无状态的。
# ------------------------------------------------------------------------------

# 导入必要的类型和接口
from typing import List, Dict, Optional
from loguru import Logger
import itertools # <-- 导入以方便生成块对

# 核心接口 (必须遵循)
from kd_tool.core.interfaces import StageInterface
from kd_tool.storage.storage_interface import StorageInterface

# 数据传输对象 (必须使用 v4 定义)
from kd_tool.schemas.dtos import (
    PipelineContextDTO,
    ContentBlockDTO,
    AnalysisResultDTO,
    MD5AnalysisStageSettings # <-- 确保使用重命名后的 Settings
)
from kd_tool.schemas.enums import AnalysisType

# 自定义错误 (必须使用)
from .errors import MD5InputError, MD5CalculationError

class MD5AnalysisStage(StageInterface):
    """
    MD5 分析阶段实现 (计算与比较合并)。

    负责计算内容块的 MD5 哈希值，找出重复项，并生成 AnalysisResultDTO。
    """

    def __init__(
        self,
        logger: Logger,
        storage: StorageInterface, # 尽管此阶段可能不直接写存储，但保留接口以备未来或读取需要
        settings: MD5AnalysisStageSettings # <-- 使用正确的 Settings 类型
    ) -> None:
        """
        构造函数 - 严格执行依赖注入。
        """
        self._logger = logger.bind(stage="MD5AnalysisStage")
        self._storage = storage
        self._settings = settings
        self._logger.info("MD5AnalysisStage (Combined) initialized.")

    def process(self, context: PipelineContextDTO) -> PipelineContextDTO:
        """
        执行 MD5 计算与比较流程。
        """
        task_id = context.get_task_id()
        logger = self._logger.bind(task_id=task_id)

        logger.info("Starting combined MD5 analysis process...")

        if not self._settings.enabled:
            logger.warning("MD5AnalysisStage is disabled. Skipping.")
            return context

        # 架构说明: 从 Context 获取待处理块。
        # 我们假设 get_content_blocks_for_analysis 返回所有需要检查的块。
        # 这里我们不再传入 AnalysisType.MD5，因为此 Stage 自身就是 MD5。
        # 或者，我们在这里直接获取所有块，并在内部判断是否需要处理。
        # 为简单起见，我们假设获取所有块，并在下面检查是否已有 MD5。
        content_blocks_to_check = list(context.content_blocks.values())

        if not content_blocks_to_check:
            logger.info("No content blocks found in context. Skipping.")
            return context

        logger.info(f"Found {len(content_blocks_to_check)} blocks to check for MD5.")

        md5_map: Dict[str, List[str]] = {}
        processed_count = 0
        error_count = 0

        # --- 1. 计算阶段 ---
        logger.debug("Phase 1: Calculating MD5 hashes...")
        for block in content_blocks_to_check:
            try:
                # 架构约束: 检查是否已有 MD5 且不需要强制重算 (如果settings支持)
                # (为简化，我们总是计算，但可以添加此逻辑)
                # if block.text_hash_md5 and not self._settings.force_recalculate:
                #    logger.trace(f"Block {block.block_id} already has MD5. Adding to map.")
                #    md5_map.setdefault(block.text_hash_md5, []).append(block.block_id)
                #    processed_count += 1
                #    continue

                logger.trace(f"Calculating MD5 for block_id: {block.block_id}...")

                # 架构约束: 必须检查输入有效性
                if not block.analysis_text:
                    raise MD5InputError(block.block_id, "Analysis text is empty or None.")

                # 架构约束: 必须调用辅助方法计算，并处理计算错误
                md5_hash = self._calculate_block_md5(block.analysis_text, block.block_id)

                # 架构约束: 更新 Context 中的 DTO
                # Pydantic 模型是不可变的，但 PipelineContextDTO 中的字典是可变的。
                # 最好的方式是获取块，更新，然后放回。
                # 或者，如果 DTO 是可变的，可以直接修改。
                # 为简单起见，我们假设可以直接修改或通过 context 方法更新。
                # **重要**: 在实际实现中，需要确保 Pydantic 模型的更新是正确的。
                # 我们这里只更新其 text_hash_md5 字段。
                context.content_blocks[block.block_id].text_hash_md5 = md5_hash

                # 架构约束: 构建 MD5 映射表
                md5_map.setdefault(md5_hash, []).append(block.block_id)
                processed_count += 1

            except (MD5InputError, MD5CalculationError) as e:
                logger.error(f"MD5 processing error for block {block.block_id}: {e}")
                context.add_error(e)
                error_count += 1
            except Exception as e:
                # 捕获其他意外错误
                logger.exception(f"Unexpected error processing block {block.block_id} for MD5.")
                context.add_error(MD5CalculationError(block.block_id, original_exception=e))
                error_count += 1

        logger.info(f"Phase 1 finished. Calculated MD5 for {processed_count} blocks, Errors: {error_count}.")

        # --- 2. 比较与结果生成阶段 ---
        logger.debug("Phase 2: Finding duplicates and generating results...")
        duplicate_groups_found = 0
        results_generated = 0

        for md5_hash, block_ids in md5_map.items():
            if len(block_ids) > 1:
                duplicate_groups_found += 1
                logger.debug(f"Found duplicate group (MD5: {md5_hash}): {block_ids}")

                # 架构约束: 必须为所有块对生成 AnalysisResultDTO
                for block_id_1, block_id_2 in itertools.combinations(block_ids, 2):
                    try:
                        # 架构约束: 使用 v4 DTO 定义创建结果
                        analysis_result = AnalysisResultDTO(
                            block_id_1=block_id_1,
                            block_id_2=block_id_2,
                            analysis_type=AnalysisType.MD5, # <-- 使用正确的 Enum
                            score=1.0, # MD5 匹配分数总是 1.0
                            details={"md5_hash": md5_hash} # 添加额外信息
                        )
                        # 架构约束: 将结果添加到 Context
                        context.add_analysis_result(analysis_result)
                        results_generated += 1
                    except Exception as e:
                        # 捕获创建或添加 DTO 时可能发生的错误
                        logger.error(f"Error generating MD5 result for pair ({block_id_1}, {block_id_2}): {e}")
                        context.add_error(MD5AnalysisError(f"Failed to generate result for pair ({block_id_1}, {block_id_2})", original_exception=e))
                        error_count += 1

        logger.info(f"Phase 2 finished. Found {duplicate_groups_found} duplicate groups, generated {results_generated} results.")
        logger.info(f"Combined MD5 analysis process finished. Total errors: {error_count}.")

        return context

    def _calculate_block_md5(self, text: str, block_id: str) -> str:
        """
        计算文本的 MD5 哈希值 (伪代码)。

        架构说明:
            - **此处无需展示真实的 hashlib 实现细节**。
            - **coding 阶段要求**: 必须使用 'utf-8' 编码。
            - **coding 阶段要求**: 必须捕获 `UnicodeEncodeError` 并抛出 `MD5CalculationError`。
        """
        try:
            # import hashlib
            # md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            md5_hash = f"dummy_md5_for_{block_id}" # 伪代码示例
            return md5_hash
        except UnicodeEncodeError as e:
            # 架构约束: 必须捕获并包装为自定义异常
            raise MD5CalculationError(block_id, original_exception=e)
        except Exception as e:
            # 捕获其他可能的未知错误
            raise MD5CalculationError(block_id, original_exception=e)

```