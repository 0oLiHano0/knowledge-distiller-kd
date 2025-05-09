import re
import hashlib
import logging
from typing import List, Optional, Dict, Any

from knowledge_distiller_kd.core.models import BlockDTO, BlockType, DecisionType
from knowledge_distiller_kd.core.utils import normalize_text_for_analysis


def _extract_language(line: str) -> Optional[str]:
    """从围栏起始行中提取语言标识"""
    match = re.match(r"```(\w+)", line.strip())
    return match.group(1) if match else None


def _is_start_fence(line: str) -> bool:
    """判断是否为起始围栏行（含语言或不完整围栏）"""
    stripped = line.strip()
    return stripped.startswith("```") and stripped != "```"


def _is_end_fence(line: str) -> bool:
    """判断是否为结束围栏行"""
    return line.strip() == "```"


def _is_complete_fence(line: str) -> bool:
    """判断是否为自包含完整围栏代码块开闭行"""
    stripped = line.strip()
    return stripped.startswith("```") and stripped.endswith("```") and stripped != "```"


def _assemble_code_body(blocks: List[BlockDTO]) -> str:
    """聚合当前片段中所有代码行，去除首尾围栏"""
    merged: List[str] = []
    for idx, blk in enumerate(blocks):
        lines = blk.text_content.splitlines(keepends=True)
        # 去掉起始围栏
        if idx == 0 and _is_start_fence(lines[0]):
            merged.extend(lines[1:])
        # 去掉尾围栏
        elif idx == len(blocks) - 1 and _is_end_fence(lines[-1]):
            merged.extend(lines[:-1])
        else:
            merged.extend(lines)
    return "".join(merged)


def _create_merged_block(
    segment: List[BlockDTO], max_gap: int, logger: logging.Logger
) -> BlockDTO:
    """创建一个合并后的 BlockDTO"""
    first = segment[0]
    file_id = first.file_id

    # 特殊场景：仅有起始和结束围栏，无实际代码，直接返回空内容块
    if len(segment) == 2 and all(blk.text_content.strip() == "```" for blk in segment):
        raw = (file_id or "")
        new_id = hashlib.md5(raw.encode()).hexdigest()
        merged = BlockDTO(
            block_id=new_id,
            file_id=file_id,
            block_type=BlockType.CODE_MERGED,
            text_content="",
            analysis_text="",
            char_count=0,
            token_count=sum(b.token_count for b in segment),
            metadata=dict(first.metadata or {}),
            kd_processing_status=DecisionType.KEEP
        )
        logger.debug(f"Merged empty fences {[b.block_id for b in segment]} into {new_id}")
        for b in segment:
            b.kd_processing_status = DecisionType.DELETE
            b.duplicate_of_block_id = new_id
        return merged

    # 正常合并逻辑
    first_line = first.text_content.splitlines()[0]
    language = _extract_language(first_line)
    code_body = _assemble_code_body(segment)
    # 根据 max_gap 决定是否保留围栏
    if max_gap == 0:
        start_fence = f"```{language}" if language else "```"
        full_text = f"{start_fence}\n{code_body.rstrip()}\n```"
    else:
        full_text = code_body
    # 生成分析文本
    try:
        analysis = normalize_text_for_analysis(code_body)
    except Exception:
        analysis = code_body
    # 生成新的 block_id
    raw = (file_id or "") + full_text
    new_id = hashlib.md5(raw.encode()).hexdigest()
    # 构建 metadata
    metadata = dict(first.metadata or {})
    if language:
        metadata["language"] = language
    # 创建合并块
    merged = BlockDTO(
        block_id=new_id,
        file_id=file_id,
        block_type=BlockType.CODE_MERGED,
        text_content=full_text,
        analysis_text=analysis,
        char_count=len(full_text),
        token_count=sum(b.token_count for b in segment),
        metadata=metadata,
        kd_processing_status=DecisionType.KEEP
    )
    logger.debug(f"Merged code blocks {[b.block_id for b in segment]} into {new_id}")
    for b in segment:
        b.kd_processing_status = DecisionType.DELETE
        b.duplicate_of_block_id = new_id
    return merged


def merge_code_blocks(
    current_blocks: List[BlockDTO],
    config: Dict[str, Any],
    logger: logging.Logger
) -> List[BlockDTO]:
    """将 unstructured 拆分的 Markdown 代码块进行合并"""
    processed: List[BlockDTO] = []
    segment: List[BlockDTO] = []
    in_fenced = False
    current_file = None
    non_code = 0
    max_gap = config.get(
        "processing.merging.max_consecutive_non_code_lines_to_break_merge", 1
    )

    def _flush():
        nonlocal in_fenced, non_code
        if not segment:
            return
        processed.append(_create_merged_block(segment, max_gap, logger))
        segment.clear()
        in_fenced = False
        non_code = 0

    for idx, block in enumerate(current_blocks):
        text = block.text_content or ""
        stripped = text.strip()
        next_block = current_blocks[idx + 1] if idx + 1 < len(current_blocks) else None

        # 文件变更时先 flush
        if block.file_id != current_file:
            if in_fenced:
                _flush()
            in_fenced = False
            non_code = 0
            current_file = block.file_id

        if not in_fenced:
            # 自包含完整围栏块，不合并
            if block.block_type == BlockType.CODE and _is_complete_fence(text):
                processed.append(block)
            # 纯围栏 start and next is pure fence => 空代码块
            elif (
                block.block_type == BlockType.CODE
                and stripped == "```"
                and next_block
                and next_block.text_content.strip() == "```"
            ):
                in_fenced = True
                segment.append(block)
            # 起始围栏（含语言标识）
            elif block.block_type == BlockType.CODE and _is_start_fence(text):
                in_fenced = True
                segment.append(block)
            else:
                processed.append(block)
        else:
            if block.block_type == BlockType.CODE and _is_end_fence(text):
                segment.append(block)
                _flush()
            elif block.block_type != BlockType.CODE:
                non_code += 1
                if non_code > max_gap:
                    processed.extend(segment)
                    processed.append(block)
                    segment.clear()
                    in_fenced = False
                    non_code = 0
                else:
                    segment.append(block)
            else:
                segment.append(block)

    # 文件末尾未闭合
    if in_fenced and segment:
        logger.warning(
            "Unclosed code block detected at end of file. Flushing accumulated fragments."
        )
        _flush()

    return processed
