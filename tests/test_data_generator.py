"""
测试数据生成器模块。

此模块用于生成测试数据，包括：
1. MD5重复内容
2. 语义重复内容
3. 正常内容
4. 决策数据
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

class DataGenerator:
    """
    测试数据生成器类。

    该类负责：
    1. 生成MD5重复内容
    2. 生成语义重复内容
    3. 生成正常内容
    4. 生成决策数据
    """

    def __init__(self) -> None:
        """
        初始化测试数据生成器。
        """
        self._md5_duplicates = []
        self._semantic_duplicates = []
        self._normal_content = []
        self._decisions = {}

    def generate_md5_duplicates(self, output_dir: Optional[Union[str, Path]] = None, count: int = 3) -> List[Tuple[str, str]]:
        """
        生成MD5重复内容。

        Args:
            output_dir: 输出目录路径
            count: 生成的文件数量

        Returns:
            List[Tuple[str, str]]: 文件名和内容的列表
        """
        if not self._md5_duplicates:
            # 生成基础内容
            base_content = "这是一个测试内容，用于验证MD5重复检测功能。"
            # 生成重复内容
            for i in range(count):
                filename = f"md5_duplicate_{i+1}.md"
                self._md5_duplicates.append((filename, base_content))

        # 如果指定了输出目录，写入文件
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in self._md5_duplicates:
                file_path = output_dir / filename
                file_path.write_text(content, encoding='utf-8')

        return self._md5_duplicates

    def generate_semantic_duplicates(self, output_dir: Optional[Union[str, Path]] = None, count: int = 3) -> List[Tuple[str, str]]:
        """
        生成语义重复内容。

        Args:
            output_dir: 输出目录路径
            count: 生成的文件数量

        Returns:
            List[Tuple[str, str]]: 文件名和内容的列表
        """
        if not self._semantic_duplicates:
            # 生成基础内容
            base_contents = [
                "这是一个测试内容，用于验证语义重复检测功能。",
                "这是一个测试文档，用于验证语义相似度检测功能。",
                "这是一个测试文本，用于验证内容相似度检测功能。"
            ]
            # 生成重复内容
            for i in range(count):
                filename = f"semantic_duplicate_{i+1}.md"
                content = base_contents[i % len(base_contents)]
                self._semantic_duplicates.append((filename, content))

        # 如果指定了输出目录，写入文件
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in self._semantic_duplicates:
                file_path = output_dir / filename
                file_path.write_text(content, encoding='utf-8')

        return self._semantic_duplicates

    def generate_normal_content(self, output_dir: Optional[Union[str, Path]] = None, count: int = 3) -> List[Tuple[str, str]]:
        """
        生成正常内容。

        Args:
            output_dir: 输出目录路径
            count: 生成的文件数量

        Returns:
            List[Tuple[str, str]]: 文件名和内容的列表
        """
        if not self._normal_content:
            # 生成正常内容
            for i in range(count):
                filename = f"normal_{i+1}.md"
                content = f"这是第{i+1}个正常内容，用于测试。"
                self._normal_content.append((filename, content))

        # 如果指定了输出目录，写入文件
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in self._normal_content:
                file_path = output_dir / filename
                file_path.write_text(content, encoding='utf-8')

        return self._normal_content

    def generate_test_files(self, output_dir: Union[str, Path], md5_count: int = 3, semantic_count: int = 3, normal_count: int = 3) -> None:
        """
        生成测试文件。

        Args:
            output_dir: 输出目录路径
            md5_count: MD5重复文件数量
            semantic_count: 语义重复文件数量
            normal_count: 正常文件数量
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成所有类型的文件
        self.generate_md5_duplicates(output_dir, md5_count)
        self.generate_semantic_duplicates(output_dir, semantic_count)
        self.generate_normal_content(output_dir, normal_count)

    def generate_decisions(self, md5_duplicates: Optional[List[Tuple[str, str]]] = None,
                         semantic_duplicates: Optional[List[Tuple[str, str]]] = None) -> Dict[str, str]:
        """
        生成决策数据。

        Args:
            md5_duplicates: MD5重复内容列表
            semantic_duplicates: 语义重复内容列表

        Returns:
            Dict[str, str]: 决策字典
        """
        decisions = {}

        # 处理MD5重复
        if md5_duplicates:
            for i, (filename, _) in enumerate(md5_duplicates):
                key = f"{filename}::0::paragraph"
                decisions[key] = "keep" if i == 0 else "delete"

        # 处理语义重复
        if semantic_duplicates:
            for i, (filename, _) in enumerate(semantic_duplicates):
                key = f"{filename}::0::paragraph"
                decisions[key] = "keep" if i == 0 else "delete"

        return decisions

# 创建数据生成器实例
test_data_generator = DataGenerator() 