# KD_Tool_CLI/setup.py

from setuptools import setup, find_packages
import pathlib # 推荐导入 pathlib 用于读取 README

# 获取项目根目录
here = pathlib.Path(__file__).parent.resolve()

# 获取 README.md 的内容作为 long_description
# (可选，但推荐) 确保你的 README.md 是 UTF-8 编码
try:
    long_description = (here / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = "Knowledge Distiller (KD) Tool: Detects and manages duplicate content in Markdown files."

setup(
    name="knowledge_distiller_kd", # 项目名称，import 时使用
    version="1.0.0", # 项目版本号
    description="A tool to detect and manage duplicate content in Markdown files.", # 项目简短描述
    long_description=long_description, # 从 README 读取的长描述
    long_description_content_type="text/markdown", # 长描述的格式
    url="https://github.com/your_username/your_repo_name", # 项目的 URL (可选，替换成你的)
    author="Your Name or Team", # 作者名字 (可选，替换成你的)
    author_email="your.email@example.com", # 作者邮箱 (可选，替换成你的)
    # 可以添加 classifiers 来对项目进行分类
    classifiers=[
        "Development Status :: 3 - Alpha", # 开发状态 (例如: 3 - Alpha, 4 - Beta, 5 - Production/Stable)
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Text Processing :: Markup :: Markdown",
        "License :: OSI Approved :: MIT License", # 假设是 MIT 许可 (可选，替换成你的)
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="markdown, duplicate content, knowledge management, semantic search", # 项目关键字 (可选)
    packages=find_packages(exclude=["tests", "docs"]), # 自动查找项目中的包，排除测试和文档目录
    python_requires=">=3.8", # 指定 Python 版本要求
    install_requires=[ # 项目运行所需的依赖库
        # 保持这里的列表与 requirements.txt 同步或稍微宽松
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.0",
        "torch>=2.0.0", # 注意 torch 可能很大，根据需要指定 cpu 或 gpu 版本
        "tqdm>=4.65.0",
        # "pathlib>=1.0.1", # pathlib 在 Python 3.4+ 是标准库，通常不需要列出
        # "typing>=3.7.4", # typing 在 Python 3.5+ 是标准库，通常不需要列出
        "unstructured[md]>=0.10.0", # 使用 unstructured 并指定 markdown extras
        "PyYAML>=6.0.0",
        "colorama>=0.4.6", # 如果 UI 使用了颜色
        # 其他你可能添加的依赖...
    ],
    # 可选的额外依赖组 (例如，用于开发的依赖)
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0", "pytest-mock>=3.0", "flake8", "black"],
        "test": ["pytest>=7.0", "pytest-cov>=4.0", "pytest-mock>=3.0"],
    },

    # ==================== 添加 Entry Point ====================
    entry_points={
        "console_scripts": [
            # 定义命令行入口点
            # 格式是: "命令名称 = 包名.模块名:函数名"
            "kd-tool = knowledge_distiller_kd.cli:main",
        ],
    },
    # =========================================================

    # 其他可选元数据
    # project_urls={ # 可选的项目相关链接
    #     "Bug Reports": "https://github.com/your_username/your_repo_name/issues",
    #     "Source": "https://github.com/your_username/your_repo_name/",
    # },
)