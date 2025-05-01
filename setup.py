from setuptools import setup, find_packages

setup(
    name="knowledge_distiller_kd",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "tqdm>=4.65.0",
        "pathlib>=1.0.1",
        "typing>=3.7.4",
        "markdown>=3.4.0",
        "PyYAML>=6.0.0",
        "colorama>=0.4.6",
    ],
    python_requires=">=3.8",
) 