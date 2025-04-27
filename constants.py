# [DEPENDENCIES]
# 1. Python Standard Library: pathlib

from pathlib import Path

# --- 常量定义 ---
SEMANTIC_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
# 默认相似度阈值，可以在命令行覆盖
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DECISION_KEY_SEPARATOR = "::"
DEFAULT_OUTPUT_SUFFIX = "_deduped"
# 默认决策文件名
DEFAULT_DECISION_FILENAME = "kd_decisions.json"

# --- 其他可能需要的常量 ---
# (可以根据需要添加)

