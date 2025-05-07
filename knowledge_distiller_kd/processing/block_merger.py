# knowledge_distiller_kd/processing/block_merger.py
"""
合并由 unstructured 解析而碎片化的 Markdown 围栏代码块。
"""
import re
import logging
import hashlib
from typing import List, Optional, Dict, Any

from knowledge_distiller_kd.core.models import BlockDTO, BlockType, DecisionType
from knowledge_distiller_kd.core.utils import normalize_text_for_analysis

def merge_code_blocks(
    current_blocks: List[BlockDTO],
    config: Dict[str, Any],
    logger: logging.Logger
) -> List[BlockDTO]:
    """
    合并同一文件中由 unstructured 解析碎片化的 Markdown 围栏代码块。
    """
    processed_blocks: List[BlockDTO] = []
    current_segment: List[BlockDTO] = []
    in_fenced = False
    current_file_id: Optional[str] = None
    non_code_count = 0
    max_gap = config.get(
        "processing.merging.max_consecutive_non_code_lines_to_break_merge", 1
    )

    def flush_segment():
        nonlocal in_fenced, non_code_count
        if not current_segment:
            return
        first = current_segment[0]
        file_id = first.file_id
        first_line = first.text_content.strip().splitlines()[0] if first.text_content else ""
        match = re.match(r"```(\w+)", first_line)
        language = match.group(1) if match else None
        merged_lines: List[str] = []
        for idx, blk in enumerate(current_segment):
            lines = blk.text_content.splitlines(keepends=True)
            if idx == 0:
                if lines and lines[0].strip().startswith("```"):
                    merged_lines.extend(lines[1:])
                else:
                    merged_lines.extend(lines)
            elif idx == len(current_segment) - 1:
                if lines and lines[-1].strip() == "```":
                    merged_lines.extend(lines[:-1])
                else:
                    merged_lines.extend(lines)
            else:
                merged_lines.extend(lines)
        merged_text = "".join(merged_lines)
        try:
            analysis_text = normalize_text_for_analysis(merged_text)
        except Exception:
            analysis_text = merged_text
        raw = (file_id or "") + merged_text
        new_id = hashlib.md5(raw.encode()).hexdigest()
        metadata = dict(first.metadata or {})
        metadata["language"] = language
        char_count = len(merged_text)
        token_count = sum(b.token_count for b in current_segment)
        merged_block = BlockDTO(
            block_id=new_id,
            file_id=file_id,
            block_type=BlockType.CODE,
            text_content=merged_text,
            analysis_text=analysis_text,
            char_count=char_count,
            token_count=token_count,
            metadata=metadata,
            kd_processing_status=DecisionType.UNDECIDED
        )
        processed_blocks.append(merged_block)
        for blk in current_segment:
            blk.kd_processing_status = DecisionType.DELETE
            blk.duplicate_of_block_id = new_id
        logger.debug(
            f"Merged code blocks {[b.block_id for b in current_segment]} into {new_id}"
        )
        current_segment.clear()
        in_fenced = False
        non_code_count = 0

    i = 0
    length = len(current_blocks)
    while i < length:
        block = current_blocks[i]
        stripped = block.text_content.strip()
        next_block = current_blocks[i + 1] if i + 1 < length else None

        # 文件切换
        if block.file_id != current_file_id:
            if in_fenced:
                flush_segment()
            in_fenced = False
            non_code_count = 0
            current_file_id = block.file_id

        if not in_fenced:
            # 自包含的完整围栏块
            if (
                block.block_type == BlockType.CODE
                and stripped.startswith("```")
                and stripped.endswith("```")
                and stripped != "```"
            ):
                processed_blocks.append(block)
            # 起始围栏：带语言标识
            elif (
                block.block_type == BlockType.CODE
                and stripped.startswith("```")
                and stripped != "```"
            ):
                in_fenced = True
                current_segment.append(block)
            # 纯围栏开始：只有在后面有纯围栏闭合时
            elif (
                block.block_type == BlockType.CODE
                and stripped == "```"
                and next_block
                and next_block.text_content.strip() == "```"
            ):
                in_fenced = True
                current_segment.append(block)
            else:
                processed_blocks.append(block)
        else:
            # 结束围栏
            if block.block_type == BlockType.CODE and stripped == "```":
                current_segment.append(block)
                flush_segment()
            # 非代码行
            elif block.block_type != BlockType.CODE:
                non_code_count += 1
                if non_code_count > max_gap:
                    # 中断合并：原样输出累积的
                    for blk in current_segment:
                        processed_blocks.append(blk)
                    processed_blocks.append(block)
                    current_segment.clear()
                    in_fenced = False
                    non_code_count = 0
                else:
                    current_segment.append(block)
            # 普通代码行
            else:
                current_segment.append(block)

        i += 1

    # 文件末尾未闭合的围栏
    if in_fenced and current_segment:
        logger.warning(
            "Unclosed code block detected at end of file. Flushing accumulated fragments."
        )
        flush_segment()

    return processed_blocks
