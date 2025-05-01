# 文档预处理模块设计文档

## 1. 模块概述

### 1.1 目标
文档预处理模块是KD Tool的核心组件之一，负责将不同格式的文档转换为标准化的内容块(ContentBlock)，为后续的MD5和语义分析提供统一的数据基础。本模块将利用Unstructured库来处理底层的文档解析，优先支持Markdown(.md)和Microsoft Word(.docx)格式。

### 1.2 技术选型
1. 核心依赖：Unstructured库 (https://github.com/Unstructured-IO/unstructured.git)
   - 文档解析：使用`unstructured.partition_md`和`unstructured.partition_docx`
   - 元素类型：依赖Unstructured输出的元素类型(如Title, NarrativeText, ListItem, CodeSnippet, Table等)
   - 元数据：利用Element.metadata提供的元数据(如filename, filetype, page_number等)

2. 必要依赖(根据支持格式安装)：
   - 基础：unstructured
   - Markdown支持：通常包含在基础库中
   - Word支持：`pip install "unstructured[docx]"`(会安装python-docx, lxml等)
   - 类型提示：typing, typing_extensions

### 1.3 在KD Tool中的位置
- 输入：原始文档文件路径列表
- 输出：一个字典，键是文件路径，值是该文件对应的ContentBlock对象列表
- 调用者：KDToolCLI中的run_analysis流程
- 下游：MD5Analyzer和SemanticAnalyzer(它们将使用ContentBlock.analysis_text)

## 2. 核心类设计

### 2.1 ContentBlock（内容块类）
```python
class ContentBlock:
    """内容块类，封装Unstructured的Element对象，并提供标准化文本"""
    def __init__(self, element: Element, file_path: str):
        """
        初始化ContentBlock

        Args:
            element (Element): 来自Unstructured解析结果的元素对象
            file_path (str): 该元素所属的原始文件路径
        """
        if not isinstance(element, Element):
            raise TypeError("element must be an instance of unstructured.documents.elements.Element")

        self.element: Element = element      # 保留原始Unstructured元素
        self.file_path: str = file_path      # 来源文件路径
        self.block_id: str = str(element.id) # 使用Unstructured元素的ID作为唯一标识符
        self.analysis_text: str = self._normalize_text() # 生成用于分析的标准化文本

    def _normalize_text(self) -> str:
        """生成用于分析的标准化文本，去除格式标记，保留核心文本内容"""
        text = self.element.text
        element_type = type(self.element).__name__

        # 1. 移除首尾及中间多余的空白字符
        text = ' '.join(text.split())

        # 2. 根据类型进行特定处理
        if element_type == 'CodeSnippet':
            return self.element.text.strip()
        elif element_type == 'Table':
            pass # 继续通用处理
        else:
            # 移除常见的Markdown格式标记
            text = re.sub(r'[`*_~]', '', text)  # 移除代码标记, 粗体/斜体/删除线标记
            text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # 提取链接文本，移除URL
            text = re.sub(r'^#{1,6}\s+', '', text)  # 移除行首的标题标记
            text = re.sub(r'^\s*[-*+]\s+', '', text) # 移除列表项标记
            text = re.sub(r'^\s*>\s?', '', text) # 移除引用标记

        return text.strip()

    @property
    def original_text(self) -> str:
        """获取原始文本内容"""
        return self.element.text

    @property
    def block_type(self) -> str:
        """获取块类型(来自Unstructured Element的类名)"""
        return type(self.element).__name__

    @property
    def metadata(self) -> Dict:
        """获取一个包含关键元数据的字典"""
        return {
            'file_path': self.file_path,
            'block_id': self.block_id,
            'block_type': self.block_type,
            'original_text': self.original_text,
            'analysis_text': self.analysis_text
        }

    def __repr__(self) -> str:
        """提供一个简洁的字符串表示，方便调试"""
        return f"ContentBlock(id={self.block_id}, type={self.block_type}, file='{Path(self.file_path).name}')"
```

## 3. 处理流程

### 3.1 文件处理函数
```python
def process_file(file_path: str) -> List[ContentBlock]:
    """处理单个文件，使用Unstructured解析并转换为ContentBlock列表"""
    logger.debug(f"Processing file: {file_path}")
    content_blocks = []
    try:
        # 使用Unstructured的自动分区功能
        elements = partition(filename=file_path, strategy="fast")

        if not elements:
            logger.warning(f"No elements extracted from file: {file_path}")
            return []

        # 将解析出的Element转换为ContentBlock
        for element in elements:
            try:
                content_blocks.append(ContentBlock(element, file_path))
            except TypeError as te:
                logger.error(f"Skipping element due to TypeError in {file_path}: {te}")
            except Exception as cb_e:
                logger.error(f"Skipping element due to error in {file_path}: {cb_e}")

        return content_blocks

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except ValueError as ve:
        logger.error(f"Error processing file {file_path}: {ve}")
        return []
    except ImportError as ie:
        logger.error(f"Missing dependency for {file_path}: {ie}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")
        return []

def process_directory(dir_path: str, file_types: List[str] = ['.md', '.docx'], recursive: bool = False) -> Dict[str, List[ContentBlock]]:
    """处理指定目录下的所有支持类型的文件"""
    logger.info(f"Processing directory: {dir_path}")
    results: Dict[str, List[ContentBlock]] = {}
    p_dir = Path(dir_path)

    if not p_dir.is_dir():
        logger.error(f"Invalid directory: {dir_path}")
        return results

    file_iterator = p_dir.rglob("*") if recursive else p_dir.glob("*")
    processed_count = error_count = skipped_count = 0

    try:
        for file_path_obj in file_iterator:
            if file_path_obj.is_file():
                ext = file_path_obj.suffix.lower()
                if ext in file_types:
                    blocks = process_file(str(file_path_obj))
                    if blocks:
                        results[str(file_path_obj)] = blocks
                        processed_count += 1
                    else:
                        error_count += 1
                else:
                    skipped_count += 1

    except Exception as e:
        logger.error(f"Error processing directory {dir_path}: {e}")

    logger.info(f"Processed: {processed_count}, Errors: {error_count}, Skipped: {skipped_count}")
    return results
```

## 4. 配置设计

### 4.1 命令行参数
```python
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='文档预处理工具')
    
    # 基本参数
    parser.add_argument('--input', '-i', required=True,
                      help='输入目录路径')
    parser.add_argument('--output', '-o', required=True,
                      help='输出目录路径')
    
    # 处理选项
    parser.add_argument('--recursive', '-r', action='store_true',
                      help='是否递归处理子目录')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='是否显示详细日志')
    
    # 文件类型
    parser.add_argument('--types', '-t', nargs='+', default=['md', 'docx'],
                      help='支持的文件类型（扩展名）')
    
    return parser.parse_args()
```

## 5. 错误处理

### 5.1 基本错误处理
```python
import logging
from typing import List, Dict
from unstructured.documents.elements import Element

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_file(file_path: str) -> List[ContentBlock]:
    """处理单个文件"""
    try:
        # 根据文件类型选择partition函数
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.md':
            from unstructured.partition.md import partition_md
            elements = partition_md(file_path)
        elif ext == '.docx':
            from unstructured.partition.docx import partition_docx
            elements = partition_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # 创建内容块
        return [ContentBlock(elem, file_path) for elem in elements]
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []
```

## 6. 测试策略

### 6.1 单元测试
1. ContentBlock测试：
   - 测试`__init__`能否正确接收Element和file_path
   - 测试block_id是否正确生成
   - 重点测试`_normalize_text`：
     - 针对不同element.type
     - 包含各种Markdown标记的element.text
     - 验证输出的analysis_text是否符合预期

2. process_file测试：
   - 使用unittest.mock.patch模拟partition函数
   - 测试不支持的文件类型
   - 测试文件不存在的情况
   - 测试缺少依赖的情况

3. process_directory测试：
   - 使用tempfile.TemporaryDirectory创建测试环境
   - 测试非递归和递归扫描
   - 测试空目录或无效目录路径

### 6.2 集成测试
1. 创建包含Markdown和Word文件的测试目录
2. 运行KDToolCLI的run_analysis
3. 验证blocks_data包含预期数量和类型的ContentBlock
4. 检查analysis_text是否正确标准化

## 7. 实施计划

### 7.1 第一阶段：基础集成与Markdown支持
1. 实现ContentBlock类：
   - 完成`__init__`, `_normalize_text`, metadata属性
   - 添加类型提示和文档字符串

2. 实现process_file函数：
   - 处理.md文件
   - 调用unstructured.partition_md
   - 实现基本错误处理

3. 实现process_directory函数：
   - 实现目录扫描
   - 调用process_file

4. 集成到KDToolCLI：
   - 修改run_analysis
   - 修改决策相关函数
   - 修改分析器使用analysis_text

### 7.2 第二阶段：添加Word支持与测试完善
1. 添加Word处理：
   - 在process_file中添加.docx处理
   - 安装unstructured[docx]依赖

2. 完善_normalize_text：
   - 针对Word特定元素类型
   - 调整标准化逻辑

3. 完善测试：
   - 添加Word文件测试
   - 实现递归选项

### 7.3 第三阶段：后续优化与扩展
1. 支持更多格式(PDF, HTML等)
2. 优化_normalize_text规则
3. 考虑性能优化(并行处理等)
4. 完善错误处理和用户反馈 