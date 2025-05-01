"""
项目安装配置文件。
"""

from setuptools import setup, find_packages

setup(
    name="knowledge_distiller_kd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "numpy",
        "pytest",
        "pytest-cov",
        "pytest-mock",
    ],
    python_requires=">=3.8",
) 