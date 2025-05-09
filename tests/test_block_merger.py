import pytest
import logging
import hashlib

from knowledge_distiller_kd.processing.block_merger import merge_code_blocks
from knowledge_distiller_kd.core.models import BlockDTO, BlockType, DecisionType

# Helper to create a dummy logger
@pytest.fixture
def logger():
    logger = logging.getLogger("block_merger_test")
    # prevent printing to console
    logger.addHandler(logging.NullHandler())
    return logger

# Helper to create BlockDTO
def make_block(block_id, file_id, block_type, text_content, token_count=1):
    return BlockDTO(
        block_id=block_id,
        file_id=file_id,
        block_type=block_type,
        text_content=text_content,
        analysis_text=text_content,
        token_count=token_count
    )

def test_empty_list_returns_empty(logger):
    result = merge_code_blocks([], {}, logger)
    assert result == [], "Expected empty result for empty input"

def test_no_code_blocks_pass_through(logger):
    b1 = make_block("t1", "fileA", BlockType.TEXT, "just text")
    b2 = make_block("t2", "fileA", BlockType.HEADING, "# title")
    result = merge_code_blocks([b1, b2], {}, logger)
    assert result == [b1, b2], "Non-code blocks should pass through unchanged"

def test_single_complete_code_block_no_merge(logger):
    code = "```python\nprint('hello')\n```"
    b = make_block("c1", "fileA", BlockType.CODE, code)
    result = merge_code_blocks([b], {}, logger)
    assert len(result) == 1
    out = result[0]
    assert out is b, "Original block should be returned"
    assert out.kd_processing_status == DecisionType.UNDECIDED

def test_fragmented_code_block_merge(logger):
    start = make_block("c1", "fileA", BlockType.CODE, "```python")
    mid = make_block("c2", "fileA", BlockType.CODE, "print(1)")
    end = make_block("c3", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([start, mid, end], {}, logger)
    assert len(result) == 1
    merged = result[0]
    assert "```" not in merged.text_content
    assert merged.metadata.get("language") == "python"
    assert start.kd_processing_status == DecisionType.DELETE
    assert mid.kd_processing_status == DecisionType.DELETE
    assert end.kd_processing_status == DecisionType.DELETE
    assert start.duplicate_of_block_id == merged.block_id
    assert mid.duplicate_of_block_id == merged.block_id
    assert end.duplicate_of_block_id == merged.block_id

def test_multiple_independent_fragments_same_file(logger):
    s1 = make_block("s1", "fileA", BlockType.CODE, "```js")
    m1 = make_block("m1", "fileA", BlockType.CODE, "console.log('a');")
    e1 = make_block("e1", "fileA", BlockType.CODE, "```")
    s2 = make_block("s2", "fileA", BlockType.CODE, "```python")
    m2 = make_block("m2", "fileA", BlockType.CODE, "print('b')")
    e2 = make_block("e2", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s1, m1, e1, s2, m2, e2], {}, logger)
    assert len(result) == 2
    langs = [blk.metadata.get("language") for blk in result]
    assert set(langs) == {"js", "python"}

def test_fragments_different_files_not_merged(logger):
    s1 = make_block("1", "A", BlockType.CODE, "```python")
    c1 = make_block("2", "A", BlockType.CODE, "a=1")
    e1 = make_block("3", "A", BlockType.CODE, "```")
    s2 = make_block("4", "B", BlockType.CODE, "```python")
    c2 = make_block("5", "B", BlockType.CODE, "b=2")
    e2 = make_block("6", "B", BlockType.CODE, "```")
    result = merge_code_blocks([s1, c1, e1, s2, c2, e2], {}, logger)
    assert len(result) == 2
    assert result[0].file_id == "A"
    assert result[1].file_id == "B"

def test_allowed_gap_merges_across_non_code(logger):
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    t = make_block("t", "fileA", BlockType.TEXT, "# comment")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s, t, e], {}, logger)
    assert len(result) == 1
    merged = result[0]
    assert "# comment" in merged.text_content

def test_exceeding_gap_breaks_merge(logger):
    config = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 0}
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    t = make_block("t", "fileA", BlockType.TEXT, "# comment")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s, t, e], config, logger)
    assert len(result) == 3
    assert all(b.kd_processing_status == DecisionType.UNDECIDED for b in result)

def test_unclosed_fence_at_end_merges(logger, caplog):
    caplog.set_level(logging.WARNING)
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    c = make_block("c", "fileA", BlockType.CODE, "x=1")
    result = merge_code_blocks([s, c], {}, logger)
    assert len(result) == 1
    assert "Unclosed code block detected" in caplog.text

def test_empty_code_block_only_fences(logger):
    s = make_block("s", "fileA", BlockType.CODE, "```")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s, e], {}, logger)
    assert len(result) == 1
    merged = result[0]
    assert merged.text_content == ""

def test_default_config_key_missing_uses_default(logger):
    s = make_block("s", "fileA", BlockType.CODE, "```python")
    t1 = make_block("t1", "fileA", BlockType.TEXT, "a")
    t2 = make_block("t2", "fileA", BlockType.TEXT, "b")
    e = make_block("e", "fileA", BlockType.CODE, "```")
    result = merge_code_blocks([s, t1, t2, e], {}, logger)
    assert len(result) == 4
    assert [b.block_id for b in result] == ["s", "t1", "t2", "e"]


class TestBlockMergerAdvancedScenarios:

    def test_merge_realistic_tf_model_block(self):
        logger = logging.getLogger("test_logger")
        file_id = "test-file"

        blocks = [
            BlockDTO(
                block_id="b1",
                file_id=file_id,
                block_type=BlockType.CODE,
                text_content="```python import tensorflow as tf"
            ),
            BlockDTO(
                block_id="b2",
                file_id=file_id,
                block_type=BlockType.CODE,
                text_content=(
                    "def build_model(input_shape):\n"
                    "    model = tf.keras.Sequential([\n"
                    "        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),\n"
                    "        tf.keras.layers.Dense(10, activation='softmax')\n"
                    "    ])\n"
                    "    return model"
                )
            ),
            BlockDTO(
                block_id="b3",
                file_id=file_id,
                block_type=BlockType.CODE,
                text_content="```"
            ),
        ]

        config = {
            "processing.merging.max_consecutive_non_code_lines_to_break_merge": 0
        }

        result = merge_code_blocks(blocks, config, logger)
        merged_blocks = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        assert len(merged_blocks) == 1, "应合并为一个代码块"

        merged_block = merged_blocks[0]
        assert merged_block.text_content.startswith("```python"), "应以起始围栏开头"
        assert merged_block.text_content.endswith("```"), "应以结束围栏结尾"
        assert "build_model" in merged_block.text_content, "代码应包含模型函数"
        assert merged_block.kd_processing_status == DecisionType.KEEP

    def test_merge_code_with_misclassified_comment(self):
        logger = logging.getLogger("test_logger")
        file_id = "test-file"
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```python"),
            make_block("c2", file_id, BlockType.CODE, "print('Hello')"),
            make_block("c3", file_id, BlockType.TEXT, "# This is a comment"),
            make_block("c4", file_id, BlockType.CODE, "print('World')"),
            make_block("c5", file_id, BlockType.CODE, "```"),
        ]
        config = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 1}

        result = merge_code_blocks(blocks, config, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        assert len(merged) == 1, "带有被误分类注释的代码块应合并为一个"

        mb = merged[0]
        assert "# This is a comment" in mb.text_content, "注释行应包含在合并结果中"
        assert mb.kd_processing_status == DecisionType.KEEP

    def test_merge_with_list_items_in_code_block(self):
        """测试包含列表项的代码块合并"""
        logger = logging.getLogger("test_logger")
        file_id = "list-in-code"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```markdown"),
            make_block("c2", file_id, BlockType.CODE, "# Title\n"),
            make_block("c3", file_id, BlockType.CODE, "- Item 1\n- Item 2"),
            make_block("c4", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "带有列表项的代码块应合并为一个"
        assert "- Item 1" in merged[0].text_content, "列表项应保存在合并块中"
        assert merged[0].metadata.get("language") == "markdown", "应正确识别markdown语言"
    
    def test_merge_with_quote_blocks(self):
        """测试包含引用块的代码片段合并"""
        logger = logging.getLogger("test_logger")
        file_id = "quote-in-code"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```markdown"),
            make_block("c2", file_id, BlockType.TEXT, "正常文本"),
            make_block("c3", file_id, BlockType.CODE, "> 这是引用\n> 第二行引用"),
            make_block("c4", file_id, BlockType.CODE, "```"),
        ]
        
        # 设置允许合并跨越1行非代码
        config = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 1}
        result = merge_code_blocks(blocks, config, logger)
        
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        assert len(merged) == 1, "跨引用块的代码应合并"
        assert "正常文本" in merged[0].text_content, "非代码文本应包含在合并结果中"
        assert "> 这是引用" in merged[0].text_content, "引用块应包含在合并结果中"

    def test_merge_with_indented_code_and_spaces(self):
        """测试带有缩进和多余空格的代码块合并"""
        logger = logging.getLogger("test_logger")
        file_id = "indented-code"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```python  "),  # 末尾有空格
            make_block("c2", file_id, BlockType.CODE, "  def hello():\n    print('world')"),  # 缩进
            make_block("c3", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "带缩进的代码块应合并"
        assert "def hello()" in merged[0].text_content, "缩进代码应正确合并"
        assert merged[0].metadata.get("language") == "python", "应识别带末尾空格的python标签"

    def test_merge_with_multiline_language_tag(self):
        """测试多行语言标签的代码块合并"""
        logger = logging.getLogger("test_logger")
        file_id = "multiline-tag"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```\npython"),  # 语言标签在第二行
            make_block("c2", file_id, BlockType.CODE, "print('Hello')"),
            make_block("c3", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        
        # 当前设计并不支持多行语言标签，所以语言标签应为空
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        assert len(merged) == 1, "应合并代码块"
        assert "python" in merged[0].text_content, "语言标签行应作为内容保留"
        assert merged[0].metadata.get("language") is None, "语言标签不应该从第二行提取"

    def test_merge_with_language_case_variations(self):
        """测试不同大小写语言标签的识别"""
        logger = logging.getLogger("test_logger")
        file_id = "language-case"
        
        # 创建不同大小写的语言标签块
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```Python"),  # 首字母大写
            make_block("c2", file_id, BlockType.CODE, "print('Hello')"),
            make_block("c3", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "应合并代码块"
        assert merged[0].metadata.get("language") == "Python", "应保留原始大小写"

    def test_merge_with_incomplete_start_fence(self):
        """测试不完整起始围栏的处理"""
        logger = logging.getLogger("test_logger")
        file_id = "incomplete-fence"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "``"),  # 不完整围栏
            make_block("c2", file_id, BlockType.CODE, "print('Hello')"),
            make_block("c3", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        
        # 因为起始不是有效的围栏，不应合并
        assert len([b for b in result if b.block_type == BlockType.CODE_MERGED]) == 0
        assert len(result) == 3, "不应合并不完整围栏"
        assert result[0].kd_processing_status == DecisionType.UNDECIDED, "不应改变不完整围栏状态"

    def test_merge_with_unusual_code_fence(self):
        """测试不常见围栏格式的处理"""
        logger = logging.getLogger("test_logger")
        file_id = "unusual-fence"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```{python}"),  # R Markdown 风格
            make_block("c2", file_id, BlockType.CODE, "print('Hello')"),
            make_block("c3", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "应合并 R Markdown 风格围栏代码块"
        # 根据当前实现，语言标签匹配可能不会提取花括号中的值
        language = merged[0].metadata.get("language")
        assert language is None or language == "{python}", "应保留或尝试解析花括号标签"

    def test_merge_with_complex_interleaved_blocks(self):
        """测试复杂交错的块结构"""
        logger = logging.getLogger("test_logger")
        file_id = "complex-structure"
        
        # 构建复杂交错结构: 代码块1开始 -> 文本 -> 代码块1内容 -> 代码块2开始 -> 代码块2内容 -> 代码块1结束 -> 代码块2结束
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```python"),
            make_block("t1", file_id, BlockType.TEXT, "中间文本"),
            make_block("c2", file_id, BlockType.CODE, "print('First block')"),
            make_block("c3", file_id, BlockType.CODE, "```javascript"),
            make_block("c4", file_id, BlockType.CODE, "console.log('Second block')"),
            make_block("c5", file_id, BlockType.CODE, "```"),  # 第一个代码块结束
            make_block("c6", file_id, BlockType.CODE, "```"),  # 第二个代码块结束
        ]
        
        # 允许合并跨越非代码块
        config = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 1}
        result = merge_code_blocks(blocks, config, logger)
        
        # 由于复杂交错，我们应该得到部分合并的结果
        # 理想情况是识别出两个不同语言的代码块，但具体行为取决于算法处理能力
        code_blocks = [b for b in result if b.block_type in [BlockType.CODE, BlockType.CODE_MERGED]]
        assert len(code_blocks) > 0, "应至少合并部分代码块"
        
        # 检查结果是否包含两种不同语言的内容
        content = " ".join([b.text_content for b in code_blocks])
        assert "print('First block')" in content, "Python块内容应被保留"
        assert "console.log('Second block')" in content, "JavaScript块内容应被保留"

    def test_merge_blocks_with_html_tags(self):
        """测试包含HTML标签的代码块合并"""
        logger = logging.getLogger("test_logger")
        file_id = "html-in-code"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```html"),
            make_block("c2", file_id, BlockType.CODE, "<div class='container'>"),
            make_block("c3", file_id, BlockType.TEXT, "<!-- 注释 -->"),
            make_block("c4", file_id, BlockType.CODE, "</div>"),
            make_block("c5", file_id, BlockType.CODE, "```"),
        ]
        
        config = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 1}
        result = merge_code_blocks(blocks, config, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "HTML代码块应合并"
        assert "<!-- 注释 -->" in merged[0].text_content, "注释应作为中间内容保留"
        assert "<div" in merged[0].text_content and "</div>" in merged[0].text_content, "HTML标签应保留"
        assert merged[0].metadata.get("language") == "html", "应识别为html语言"

    def test_merge_extremely_fragmented_code(self):
        """测试极度碎片化的代码块合并"""
        logger = logging.getLogger("test_logger")
        file_id = "extremely-fragmented"
        
        # 创建每行都是单独一个块的极度碎片化代码
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```python"),
            make_block("c2", file_id, BlockType.CODE, "def"),
            make_block("c3", file_id, BlockType.CODE, "factorial"),
            make_block("c4", file_id, BlockType.CODE, "(n):"),
            make_block("c5", file_id, BlockType.CODE, "    if"),
            make_block("c6", file_id, BlockType.CODE, "        n <= 1:"),
            make_block("c7", file_id, BlockType.CODE, "        return 1"),
            make_block("c8", file_id, BlockType.CODE, "    return n*factorial(n-1)"),
            make_block("c9", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "极度碎片化的代码块应合并为一个"
        merged_text = merged[0].text_content
        assert "def factorial(n):" in merged_text, "函数声明应被正确合并"
        assert "return n*factorial(n-1)" in merged_text, "函数体应被正确合并"
        
        # 确认所有原始块都被标记为删除
        for i in range(1, 10):
            original = next((b for b in blocks if b.block_id == f"c{i}"), None)
            assert original and original.kd_processing_status == DecisionType.DELETE, f"原始碎片 c{i} 应被标记为删除"
            assert original.duplicate_of_block_id == merged[0].block_id, f"原始碎片 c{i} 应指向合并块"

    def test_merge_with_large_non_code_gap(self):
        """测试大段非代码间隔的处理"""
        logger = logging.getLogger("test_logger")
        file_id = "large-gap"
        
        # 创建中间有多个非代码块的场景
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```python"),
            make_block("t1", file_id, BlockType.TEXT, "段落1"),
            make_block("t2", file_id, BlockType.TEXT, "段落2"),
            make_block("t3", file_id, BlockType.TEXT, "段落3"),
            make_block("c2", file_id, BlockType.CODE, "print('Hello')"),
            make_block("c3", file_id, BlockType.CODE, "```"),
        ]
        
        # 设置不同的 max_gap 值进行测试
        config1 = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 1}
        result1 = merge_code_blocks(blocks, config1, logger)
        
        # 间隔超过1应导致不合并
        assert len([b for b in result1 if b.block_type == BlockType.CODE_MERGED]) == 0, "gap=1应不合并"
        
        # 增加 max_gap 到3应该允许合并
        config2 = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 3}
        result2 = merge_code_blocks(blocks, config2, logger)
        merged = [b for b in result2 if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "增加max_gap后应合并"
        for text in ["段落1", "段落2", "段落3"]:
            assert text in merged[0].text_content, f"非代码块 '{text}' 应包含在合并结果中"

    def test_merge_with_empty_blocks(self):
        """测试包含空内容块的合并"""
        logger = logging.getLogger("test_logger")
        file_id = "empty-blocks"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```python"),
            make_block("c2", file_id, BlockType.CODE, ""),  # 空内容块
            make_block("c3", file_id, BlockType.CODE, "print('After empty')"),
            make_block("c4", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "包含空块的代码应合并"
        assert "print('After empty')" in merged[0].text_content, "空块后内容应被保留"
        # 确认空行被正确处理
        assert merged[0].text_content.count("\n") >= 2, "空行应被保留为换行符"

    def test_merge_mixed_string_quotation_styles(self):
        """测试混合引号风格的代码合并"""
        logger = logging.getLogger("test_logger")
        file_id = "mixed-quotes"
        
        blocks = [
            make_block("c1", file_id, BlockType.CODE, "```python"),
            make_block("c2", file_id, BlockType.CODE, "s1 = 'single quotes'"),
            make_block("c3", file_id, BlockType.CODE, 's2 = "double quotes"'),
            make_block("c4", file_id, BlockType.CODE, 's3 = """triple double quotes"""'),
            make_block("c5", file_id, BlockType.CODE, "```"),
        ]
        
        result = merge_code_blocks(blocks, {}, logger)
        merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
        
        assert len(merged) == 1, "混合引号风格代码应合并"
        merged_text = merged[0].text_content
        assert "single quotes" in merged_text, "单引号字符串应保留"
        assert "double quotes" in merged_text, "双引号字符串应保留"
        assert "triple double quotes" in merged_text, "三引号字符串应保留"

def test_nested_code_blocks_in_blockquote(logger):
    """测试嵌套在引用块中的代码块合并"""
    file_id = "nested-quote"
    
    blocks = [
        make_block("q1", file_id, BlockType.TEXT, "> Here's some code:"),
        make_block("c1", file_id, BlockType.CODE, "> ```python"),
        make_block("c2", file_id, BlockType.CODE, "> print('Hello')"),
        make_block("c3", file_id, BlockType.CODE, "> ```"),
        make_block("q2", file_id, BlockType.TEXT, "> That was cool!"),
    ]
    
    config = {"processing.merging.max_consecutive_non_code_lines_to_break_merge": 1}
    result = merge_code_blocks(blocks, config, logger)
    
    # 因为引用和代码块类型不同，可能不会被完美合并，但我们至少要确保保留内容
    assert "print('Hello')" in " ".join([b.text_content for b in result]), "代码内容应被保留"
    # 检查引用块也被保留
    assert "Here's some code:" in " ".join([b.text_content for b in result]), "引用内容应被保留"
    assert "That was cool!" in " ".join([b.text_content for b in result]), "后续引用内容应被保留"

def test_code_fence_with_backslash_escape(logger):
    """测试带有反斜杠转义的代码围栏"""
    file_id = "escaped-fence"
    
    blocks = [
        make_block("c1", file_id, BlockType.CODE, "```python"),
        make_block("c2", file_id, BlockType.CODE, "print('A real \\`\\`\\` fence')"),
        make_block("c3", file_id, BlockType.CODE, "```"),
    ]
    
    result = merge_code_blocks(blocks, {}, logger)
    merged = [b for b in result if b.block_type == BlockType.CODE_MERGED]
    
    assert len(merged) == 1, "带转义字符的代码应合并"
    assert "\\`\\`\\`" in merged[0].text_content, "转义字符应保留"
