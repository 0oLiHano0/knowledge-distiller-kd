# knowledge_distiller_kd/processing/block_merger.py <--- 新路径
"""
Provides functionality to merge fragmented content blocks, especially code blocks.
"""

# --- 标准库导入 ---
import re
import logging
from typing import List

# --- 项目内部模块导入 (修改为相对导入) ---
from .document_processor import ContentBlock # ContentBlock 现在在同级目录

from ..core import constants # 使用 ..core 从 processing 上一级找到 core

# --- 第三方库导入 ---
# 导入必要的 Element 类型如果需要创建新的 ContentBlocks
from unstructured.documents.elements import CodeSnippet

# --- 函数定义开始 ---
logger = logging.getLogger(constants.LOGGER_NAME)

def merge_code_blocks(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """
    Merges fragmented code blocks (start fence, content, end fence) parsed by
    unstructured into single ContentBlock objects of type CodeSnippet.

    Relies on text pattern matching for fence detection, not element types.

    Args:
        blocks: The initial list of ContentBlock objects from document processing.

    Returns:
        A new list of ContentBlock objects with code blocks merged.
    """
    if not blocks:
        return []

    merged_blocks: List[ContentBlock] = []
    current_code_block_parts: List[ContentBlock] = []
    in_code_block = False

    # Regex to detect start fence (``` or ```language) - simplified check at start
    start_fence_prefix = "```"
    # End fence is just ``` stripped
    end_fence_text = "```"

    for i, block in enumerate(blocks):
        original_text_stripped = block.original_text.strip()

        # Determine if the current block looks like a start or end fence based on text
        # A block is a potential start fence if its stripped text starts with ```
        is_potential_start_fence = original_text_stripped.startswith(start_fence_prefix)
        # A block is an end fence if its stripped text is exactly ```
        is_end_fence = (original_text_stripped == end_fence_text)

        if not in_code_block:
            # If not in a block, and it looks like a start fence, begin code block
            if is_potential_start_fence:
                logger.debug(f"Block {block.block_id}: Detected start fence.")
                in_code_block = True
                current_code_block_parts = [block] # Start collecting
                # If this start fence is ALSO an end fence (i.e., just ```),
                # we immediately look for the next block to see if it's the end.
                # This handles the empty ``` case correctly in the next iteration.
            else:
                # Not in a code block and not a start fence, just add it
                merged_blocks.append(block)
        else: # We are inside a code block
            # Always add the current block to the parts list while inside
            current_code_block_parts.append(block)

            # Check if this block is the end fence
            if is_end_fence:
                logger.debug(f"Block {block.block_id}: Detected end fence.")
                # --- Perform the merge ---
                if len(current_code_block_parts) > 0: # Should always be true here
                    start_block = current_code_block_parts[0]
                    # Join original texts, preserving newlines between parts
                    # Use original_text for accurate reconstruction
                    merged_original_text = "\n".join(part.original_text for part in current_code_block_parts)

                    # --- Extract pure code for analysis_text ---
                    analysis_text = ""
                    temp_lines = merged_original_text.splitlines()
                    if len(temp_lines) >= 2: # Must have at least start and end fence
                        # Extract lines between the first and the last
                        code_lines = temp_lines[1:-1] # Exclude first and last lines
                        analysis_text = "\n".join(code_lines).strip() # Join remaining lines and strip
                    # If only fences (len(temp_lines) == 2), analysis_text remains ""

                    logger.debug(f"Merging {len(current_code_block_parts)} parts into one CodeSnippet.")
                    logger.debug(f"Merged original text preview: '{merged_original_text[:50]}...'")
                    logger.debug(f"Merged analysis text preview: '{analysis_text[:50]}...'")

                    # Create the new merged ContentBlock
                    # Use CodeSnippet type for the new element.
                    # Pass the merged original text.
                    merged_element = CodeSnippet(text=merged_original_text, element_id=start_block.block_id) # Use start block's ID

                    # Create ContentBlock instance. _infer_block_type and _normalize_text will run.
                    merged_block = ContentBlock(
                        element=merged_element,
                        file_path=start_block.file_path
                    )

                    # Crucially, override the analysis_text with our extracted pure code
                    merged_block.analysis_text = analysis_text
                    # Ensure the block type is CodeSnippet after potential inference/normalization
                    # The ContentBlock init should handle this if the element is CodeSnippet,
                    # but we can force it if needed (less ideal). Let's rely on init for now.
                    if merged_block.block_type != "CodeSnippet":
                         logger.warning(f"Merged block {merged_block.block_id} type was {merged_block.block_type}, forcing to CodeSnippet.")
                         # This direct assignment might be problematic depending on ContentBlock's design.
                         # A cleaner approach might be needed if ContentBlock resists this.
                         # Forcing it here for demonstration if ContentBlock's infer logic overrides.
                         merged_block.element.__class__ = CodeSnippet # Force the underlying element type

                    merged_blocks.append(merged_block)

                # Reset state after successful merge
                in_code_block = False
                current_code_block_parts = []
            # else: Block is part of code content, already added to parts list

    # Handle unclosed code block at the end of the file
    if in_code_block and current_code_block_parts:
        logger.warning(f"Unclosed code block detected at the end of processing file '{current_code_block_parts[0].file_path}'. Treating fragments as separate blocks.")
        # Add remaining parts as they were, without merging
        merged_blocks.extend(current_code_block_parts)

    return merged_blocks
