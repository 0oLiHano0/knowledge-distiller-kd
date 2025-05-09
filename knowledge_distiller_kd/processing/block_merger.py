import re
import hashlib
import logging
import functools
from typing import List, Optional, Dict, Any, Tuple, Set

from knowledge_distiller_kd.core.models import BlockDTO, BlockType, DecisionType
from knowledge_distiller_kd.core.utils import normalize_text_for_analysis

# 预编译正则表达式以提高性能
_LANGUAGE_PATTERN = re.compile(r"```(\w+)")

# 使用LRU缓存减少重复计算
@functools.lru_cache(maxsize=128)
def _extract_language(line: str) -> Optional[str]:
    """从围栏起始行中提取语言标识"""
    match = _LANGUAGE_PATTERN.match(line.strip())
    return match.group(1) if match else None

@functools.lru_cache(maxsize=64)
def _is_start_fence(line: str) -> bool:
    """判断是否为起始围栏行（含语言或不完整围栏）"""
    stripped = line.strip()
    return stripped.startswith("```") and stripped != "```"

@functools.lru_cache(maxsize=64)
def _is_end_fence(line: str) -> bool:
    """判断是否为结束围栏行"""
    return line.strip() == "```"

@functools.lru_cache(maxsize=64)
def _is_complete_fence(line: str) -> bool:
    """判断是否为自包含完整围栏代码块开闭行"""
    stripped = line.strip()
    return stripped.startswith("```") and stripped.endswith("```") and stripped != "```"

# 优化字符串处理函数
def _need_space_between(last_char: str, first_char: str) -> bool:
    """判断两个字符之间是否需要添加空格"""
    if not (last_char and first_char):
        return False
    
    return ((last_char.isalnum() and first_char.isalnum()) or
            (last_char in "_" and first_char.isalnum()) or
            (last_char.isalnum() and first_char in "_"))

def _assemble_code_body(blocks: List[BlockDTO]) -> str:
    """聚合当前片段中所有代码行，去除首尾围栏"""
    # 预分配足够大的列表以减少扩展操作
    merged = []
    last_content_ended_with_newline = True
    blocks_count = len(blocks)
    
    # 预处理数据以减少循环内计算
    content_cache = []
    for blk in blocks:
        content = blk.text_content
        # 直接计算和缓存常用值
        stripped = content.strip() if content else ""
        lines = content.splitlines(keepends=True) if content else []
        content_cache.append((content, stripped, lines))
    
    for idx, (content, stripped, lines) in enumerate(content_cache):
        # 空内容块处理
        if not stripped:
            if not last_content_ended_with_newline and merged:
                merged.append("\n")
            merged.append("\n")
            last_content_ended_with_newline = True
            continue
            
        # 没有换行的情况，预先处理
        if not lines:
            lines = [content]
        
        # 起始围栏处理
        if idx == 0 and lines and _is_start_fence(lines[0]):
            if len(lines) == 1:
                merged.append("\n")
                last_content_ended_with_newline = True
            else:
                merged.extend(lines[1:])
                last_content_ended_with_newline = lines[-1].endswith("\n") if lines else True
        
        # 结束围栏处理
        elif idx == blocks_count - 1 and lines and _is_end_fence(lines[-1]):
            if lines[:-1]:
                merged.extend(lines[:-1])
                last_content_ended_with_newline = lines[-2].endswith("\n")
            else:
                # 处理只有结束围栏的情况
                last_content_ended_with_newline = True
        
        # 中间内容处理
        else:
            # 优化空格添加逻辑
            if (not last_content_ended_with_newline and merged and
                    not content.startswith(("\n", " "))):
                if merged[-1]:
                    last_char = merged[-1][-1]
                    first_char = content[0] if content else ""
                    
                    if _need_space_between(last_char, first_char):
                        merged.append(" ")
            
            merged.extend(lines)
        
        # 更新换行符状态
        last_content_ended_with_newline = content.endswith("\n") if content else True
    
    # 合并片段并返回
    return "".join(merged)

def _create_merged_block(
    segment: List[BlockDTO], max_gap: int, logger: logging.Logger
) -> BlockDTO:
    """创建一个合并后的 BlockDTO"""
    first = segment[0]
    file_id = first.file_id
    segment_len = len(segment)
    
    # 快速路径：空代码块检测
    if segment_len == 2:
        text1 = segment[0].text_content.strip()
        text2 = segment[1].text_content.strip()
        if text1 == "```" and text2 == "```":
            # 从文件ID生成哈希而不是空字符串
            new_id = hashlib.md5((file_id or "").encode()).hexdigest()
            # 复用元数据而不是创建新字典
            metadata = dict(first.metadata or {})
            
            merged = BlockDTO(
                block_id=new_id,
                file_id=file_id,
                block_type=BlockType.CODE_MERGED,
                text_content="",
                analysis_text="",
                char_count=0,
                token_count=sum(b.token_count for b in segment),
                metadata=metadata,
                kd_processing_status=DecisionType.KEEP
            )
            
            # 在单次循环中设置所有块状态
            block_ids = []
            for b in segment:
                b.kd_processing_status = DecisionType.DELETE
                b.duplicate_of_block_id = new_id
                block_ids.append(b.block_id)
                
            logger.debug(f"Merged empty fences {block_ids} into {new_id}")
            return merged

    # 标准合并路径
    # 缓存第一行以避免多次分割
    first_lines = first.text_content.splitlines()
    first_line = first_lines[0] if first_lines else ""
    
    # 提取语言标识
    language = _extract_language(first_line)
    
    # 代码体组装
    code_body = _assemble_code_body(segment)
    
    # 围栏处理
    if max_gap == 0:
        # 优化字符串拼接
        start_fence = f"```{language or ''}"
        full_text = f"{start_fence}\n{code_body.rstrip()}\n```"
    else:
        full_text = code_body
    
    # 异常处理优化，只在normalize_text_for_analysis不是简单函数时有必要
    try:
        analysis = normalize_text_for_analysis(code_body)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Error normalizing text: {e}. Using raw code body.")
        analysis = code_body
    
    # 生成分块ID
    # 优化：对于短字符串，直接使用文件ID+内容生成哈希更高效
    raw = (file_id or "") + full_text
    new_id = hashlib.md5(raw.encode()).hexdigest()
    
    # 复用元数据
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
    
    # 优化：一次性收集所有块ID
    block_ids = []
    for b in segment:
        b.kd_processing_status = DecisionType.DELETE
        b.duplicate_of_block_id = new_id
        block_ids.append(b.block_id)
    
    logger.debug(f"Merged code blocks {block_ids} into {new_id}")
    return merged

def merge_code_blocks(
    current_blocks: List[BlockDTO],
    config: Dict[str, Any],
    logger: logging.Logger
) -> List[BlockDTO]:
    """将 unstructured 拆分的 Markdown 代码块进行合并"""
    # 如果无块可处理，快速返回
    if not current_blocks:
        return []
    
    total_blocks = len(current_blocks)
    processed = []
    segment = []
    in_fenced = False
    current_file = None
    non_code = 0
    
    # 配置参数优化：避免重复字典查询
    max_gap = config.get(
        "processing.merging.max_consecutive_non_code_lines_to_break_merge", 1
    )
    
    # 预处理块属性以减少循环中的计算
    # 创建(text_stripped, is_code, is_complete_fence, is_start_fence, is_end_fence)的缓存
    block_props = []
    for block in current_blocks:
        text = block.text_content or ""
        stripped = text.strip()
        is_code = block.block_type == BlockType.CODE
        
        # 只为CODE类型块计算围栏属性
        is_complete = _is_complete_fence(text) if is_code else False
        is_start = _is_start_fence(text) if is_code and not is_complete else False
        is_end = _is_end_fence(text) if is_code and not is_complete and not is_start else False
        
        block_props.append((stripped, is_code, is_complete, is_start, is_end))
    
    def _flush():
        nonlocal in_fenced, non_code
        if segment:
            processed.append(_create_merged_block(segment, max_gap, logger))
            segment.clear()
        in_fenced = False
        non_code = 0
    
    # 主处理循环
    for idx, block in enumerate(current_blocks):
        props = block_props[idx]
        stripped, is_code, is_complete, is_start, is_end = props
        
        # 获取下一个块的属性（如果有）
        next_props = block_props[idx + 1] if idx + 1 < total_blocks else None
        next_stripped = next_props[0] if next_props else None
        
        # 文件变更处理
        if block.file_id != current_file:
            if in_fenced:
                _flush()
            in_fenced = False
            non_code = 0
            current_file = block.file_id
        
        # 非围栏模式处理
        if not in_fenced:
            # 使用预计算的属性判断
            if is_code and is_complete:
                processed.append(block)
            elif is_code and stripped == "```" and next_stripped == "```":
                in_fenced = True
                segment.append(block)
            elif is_code and is_start:
                in_fenced = True
                segment.append(block)
            else:
                processed.append(block)
        # 围栏模式处理
        else:
            if is_code and is_end:
                segment.append(block)
                _flush()
            elif not is_code:
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
    
    # 处理文件末尾未闭合的情况
    if in_fenced and segment:
        logger.warning(
            "Unclosed code block detected at end of file. Flushing accumulated fragments."
        )
        _flush()
    
    return processed
