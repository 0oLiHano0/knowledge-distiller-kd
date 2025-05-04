# knowledge_distiller_kd/analysis/semantic_analyzer.py (Refactored)
"""
语义分析器模块，用于检测语义重复内容。
(Refactored to be independent of Engine/CLI)
"""

# --- 标准库导入 ---
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set, TYPE_CHECKING
import numpy as np

# --- 第三方库导入 (尝试导入) ---
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util # 别名避免与内置 util 冲突
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st_util = None # 定义以避免 NameError
    # 如果库不可用，定义一个假的 SentenceTransformer 以避免 NameError
    class SentenceTransformer: pass # type: ignore

if TYPE_CHECKING:
    # This helps type checkers understand SentenceTransformer without causing runtime errors if not installed
    from sentence_transformers import SentenceTransformer # type: ignore

# --- 项目内部模块导入 (使用相对导入) ---
from ..core.error_handler import (
    KDError, ModelError, AnalysisError, handle_error
)
from ..core.utils import (
    logger, create_decision_key, display_block_preview # 导入需要的工具函数
)
from ..core import constants
from ..processing.document_processor import ContentBlock # 导入 ContentBlock

# --- 类定义开始 ---
logger = logger # 使用 utils 中配置好的 logger

class SemanticAnalyzer:
    """
    语义分析器类，用于查找和处理语义重复内容。
    (Refactored: Independent, loads model internally, operates on input data)
    """
    # --- 所有方法定义都需要在这里开始缩进 ---

    def __init__(self,
                 similarity_threshold: float = constants.DEFAULT_SIMILARITY_THRESHOLD,
                 model_name: str = constants.DEFAULT_SEMANTIC_MODEL,
                 cache_dir: Path | str | None = None,
                 batch_size: int = constants.DEFAULT_BATCH_SIZE) -> None:
        """
        初始化语义分析器。

        Args:
            similarity_threshold: 相似度阈值（0.0 到 1.0 之间）
            model_name: 要使用的 Sentence Transformer 模型名称。
            cache_dir: 用于缓存下载模型的目录路径。如果为 None，使用默认路径。
            batch_size: 计算向量时的批处理大小。
        """
        logger.info("Initializing SemanticAnalyzer...")
        # 实例属性声明时可以带类型提示
        self.model: Optional['SentenceTransformer'] = None
        self.similarity_threshold: float = max(0.0, min(1.0, similarity_threshold))
        self.vector_cache: Dict[str, np.ndarray] = {}
        self._model_loaded: bool = False
        self.model_dimension: Optional[int] = None

        # 实例属性赋值时不带类型提示
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir if cache_dir is not None else constants.DEFAULT_CACHE_BASE_DIR).resolve()

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("`sentence-transformers` library not found. Semantic analysis will be unavailable.")


    def load_semantic_model(self) -> bool:
        """
        加载 Sentence Transformer 模型。

        Returns:
            bool: True if the model was loaded successfully or if the library is unavailable,
                  False if loading failed unexpectedly.
        """
        if self._model_loaded and self.model:
            logger.info("Semantic model '%s' already loaded.", self.model_name)
            return True
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Cannot load semantic model: `sentence-transformers` is not installed.")
            return True # Consider loading successful if library is missing, as analysis won't proceed

        try:
            model_name_to_load = self.model_name
            logger.info(f"Loading semantic model: {model_name_to_load} ... (This may take some time)")
            start_time = time.time()

            # 确认 self.cache_dir 在这里是可访问的
            logger.debug(f"DEBUG: Calling SentenceTransformer with model='{model_name_to_load}', cache_folder='{str(self.cache_dir)}'")

            # Instantiate the actual SentenceTransformer model
            self.model = SentenceTransformer(
                model_name_to_load,
                cache_folder=str(self.cache_dir) # 使用已初始化的 self.cache_dir
            )

            # Determine model dimension
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                 self.model_dimension = self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, 'encode'):
                 try:
                      dummy_embedding = self.model.encode("test")
                      if isinstance(dummy_embedding, np.ndarray):
                          self.model_dimension = dummy_embedding.shape[-1]
                      elif isinstance(dummy_embedding, list) and dummy_embedding:
                          self.model_dimension = len(dummy_embedding[0])
                      else:
                          logger.warning("Could not determine model dimension from encode output.")
                          self.model_dimension = None
                 except Exception as dim_err:
                      logger.warning(f"Could not determine model embedding dimension via encoding: {dim_err}")
                      self.model_dimension = None
            else:
                 self.model_dimension = None

            if self.model_dimension is None:
                logger.warning("Failed to get model embedding dimension information.")

            end_time = time.time()
            self._model_loaded = True
            logger.info(f"Semantic model loaded successfully. Time taken: {end_time - start_time:.2f} seconds")
            if self.model_dimension:
                logger.info(f"Model embedding dimension: {self.model_dimension}")
            return True

        except ImportError:
             logger.error("`sentence-transformers` library is missing, cannot load model.", exc_info=False)
             self._model_loaded = False
             self.model = None
             return True
        except Exception as e:
            # 捕获包括 AttributeError 在内的所有加载错误
            logger.error(f"Failed to load semantic model '{self.model_name}': {e}", exc_info=True)
            # 可以选择性地调用 handle_error，但日志已记录错误
            # handle_error(e, f"loading semantic model {self.model_name}")
            self._model_loaded = False
            self.model = None
            return False # 返回 False 表示加载失败

    def _compute_vectors(self, texts: List[str], batch_size: int = constants.DEFAULT_BATCH_SIZE) -> List[np.ndarray]:
        """
        批量计算文本的向量嵌入，使用缓存。
        (Internal method, assumes self.model is loaded)

        Args:
            texts: 要计算向量的文本列表
            batch_size: 批处理大小

        Returns:
            List[np.ndarray]: 文本向量列表 (对应输入 texts), empty arrays for errors/empty input.
        """
        # --- (这个方法的代码保持不变，确保它也被正确缩进了) ---
        if not texts or self.model is None:
            logger.warning("_compute_vectors called with no texts or model not loaded.")
            return [np.array([])] * len(texts) # Return empty arrays for all

        all_vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_compute_indices: List[int] = []
        texts_to_compute: List[str] = []
        hashes_to_compute: List[str] = []

        # Check cache
        for i, text in enumerate(texts):
            if not text: # Handle empty strings explicitly
                 all_vectors[i] = np.array([]) # Assign empty array for empty text
                 continue
            try:
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                if text_hash in self.vector_cache:
                    all_vectors[i] = self.vector_cache[text_hash]
                else:
                    texts_to_compute_indices.append(i)
                    texts_to_compute.append(text)
                    hashes_to_compute.append(text_hash)
            except Exception as hash_err:
                 logger.error(f"Error hashing text at index {i}: {hash_err}. Skipping vector computation for this text.")
                 all_vectors[i] = np.array([]) # Assign empty array on error

        # Batch compute uncached vectors
        if texts_to_compute:
            logger.info(f"Computing vectors for {len(texts_to_compute)} new texts (batch size: {batch_size})...")
            computed_vectors_list: List[np.ndarray] = []
            try:
                # Ensure model.encode returns a list or ndarray
                encode_result = self.model.encode(
                    texts_to_compute,
                    batch_size=batch_size,
                    show_progress_bar=True # Keep progress bar for user feedback
                )
                if isinstance(encode_result, np.ndarray):
                    # If ndarray, convert rows to a list of arrays
                    computed_vectors_list = [vec for vec in encode_result]
                elif isinstance(encode_result, list):
                    computed_vectors_list = encode_result
                else:
                     raise TypeError(f"Unexpected return type from model.encode: {type(encode_result)}")

            except Exception as e:
                 logger.error(f"Error during batch vector encoding: {e}", exc_info=True)
                 # Assign empty arrays to all that were supposed to be computed
                 computed_vectors_list = [np.array([])] * len(texts_to_compute)

            # Fill results and update cache
            if len(computed_vectors_list) == len(texts_to_compute):
                 for original_index, computed_vector, text_hash in zip(texts_to_compute_indices, computed_vectors_list, hashes_to_compute):
                     # Ensure vector is valid before caching
                     if isinstance(computed_vector, np.ndarray) and computed_vector.size > 0:
                         all_vectors[original_index] = computed_vector
                         self.vector_cache[text_hash] = computed_vector
                     else:
                         logger.warning(f"Invalid vector computed for text hash {text_hash}. Assigning empty array.")
                         all_vectors[original_index] = np.array([])
            else:
                 logger.error("Mismatch between computed vectors and texts to compute. Filling with empty arrays.")
                 for original_index in texts_to_compute_indices:
                      all_vectors[original_index] = np.array([])

        # Ensure all entries are processed, replacing None with empty arrays
        final_vectors: List[np.ndarray] = []
        for vec_opt in all_vectors:
            if vec_opt is None:
                final_vectors.append(np.array([]))
            else:
                final_vectors.append(vec_opt)

        return final_vectors


    def find_semantic_duplicates(
        self,
        blocks_to_analyze: List[ContentBlock]
        ) -> List[Tuple[ContentBlock, ContentBlock, float]]:
        """
        使用 Sentence Transformer 模型查找语义相似的块。
        Assumes model is loaded. Operates only on the provided blocks.

        Args:
            blocks_to_analyze: A list of ContentBlock objects to compare.
                               It's expected that these are pre-filtered (e.g., no titles,
                               no blocks already marked for deletion by MD5).

        Returns:
            List[Tuple[ContentBlock, ContentBlock, float]]: A list of tuples, where each tuple
                contains two similar ContentBlock objects and their similarity score.
                Returns an empty list if no pairs are found or if an error occurs.
        """
        # --- (这个方法的代码保持不变，确保它也被正确缩进了) ---
        logger.info(f"Starting semantic similarity analysis for {len(blocks_to_analyze)} blocks...")
        similar_pairs: List[Tuple[ContentBlock, ContentBlock, float]] = []

        if not self._model_loaded or self.model is None:
            logger.error("Semantic model not loaded, cannot perform semantic analysis.")
            return similar_pairs
        if not SENTENCE_TRANSFORMERS_AVAILABLE or st_util is None:
             logger.error("Sentence Transformers library or utils not available. Cannot perform semantic analysis.")
             return similar_pairs
        if len(blocks_to_analyze) < 2:
            logger.info("Not enough blocks (<2) provided for semantic comparison.")
            return similar_pairs

        start_time = time.time()

        try:
            # --- Step 1: Get analysis texts and compute vectors ---
            logger.info(f"Extracting texts and computing vectors for {len(blocks_to_analyze)} blocks...")
            texts_to_encode = [block.analysis_text for block in blocks_to_analyze]
            block_vectors = self._compute_vectors(texts_to_encode, self.batch_size) # Pass batch_size

            # Filter out blocks for which vector computation failed
            valid_indices = [i for i, vec in enumerate(block_vectors) if isinstance(vec, np.ndarray) and vec.size > 0]
            if len(valid_indices) < len(blocks_to_analyze):
                 failed_count = len(blocks_to_analyze) - len(valid_indices)
                 logger.warning(f"Vector computation failed for {failed_count} blocks. Proceeding with {len(valid_indices)} valid vectors.")
                 if len(valid_indices) < 2:
                      logger.error("Not enough valid vectors (<2) remaining for comparison.")
                      return similar_pairs
                 # Update blocks_to_analyze and block_vectors to only include valid ones
                 blocks_to_analyze = [blocks_to_analyze[i] for i in valid_indices]
                 block_vectors = [block_vectors[i] for i in valid_indices]

            num_valid_blocks = len(blocks_to_analyze)
            logger.info(f"Successfully computed vectors for {num_valid_blocks} blocks.")

            # --- Step 2: Compute similarity matrix and find pairs ---
            logger.info(f"Finding semantically similar pairs among {num_valid_blocks} blocks using threshold {self.similarity_threshold}...")

            # Prepare embeddings array
            embeddings = np.array(block_vectors)
            if embeddings.ndim != 2 or embeddings.shape[0] != num_valid_blocks:
                 # Check if model dimension is available for a more informative error
                 expected_dim = self.model_dimension if self.model_dimension else "unknown"
                 logger.error(f"Embeddings array has unexpected shape: {embeddings.shape}. Expected ({num_valid_blocks}, {expected_dim}). Cannot compute similarity.")
                 return similar_pairs
            if embeddings.size == 0:
                logger.info("Embeddings array is empty after filtering, cannot compute similarity.")
                return similar_pairs

            # Compute cosine similarity
            similarity_matrix = st_util.cos_sim(embeddings, embeddings) # type: ignore

            # Find pairs above threshold
            found_pairs_count = 0
            for i in range(num_valid_blocks):
                for j in range(i + 1, num_valid_blocks): # Compare distinct pairs (j > i)
                    similarity_score = similarity_matrix[i][j].item() # Get scalar value

                    if similarity_score >= self.similarity_threshold:
                        block1 = blocks_to_analyze[i]
                        block2 = blocks_to_analyze[j]
                        similar_pairs.append((block1, block2, similarity_score))
                        found_pairs_count += 1
                        logger.debug(f"Found similar pair ({similarity_score:.4f}): {block1.block_id} and {block2.block_id}")

            logger.info(f"Similarity calculation complete. Found {found_pairs_count} pairs above threshold.")

            # Sort pairs by score (descending)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)

        except Exception as e:
             logger.error(f"Error during semantic similarity calculation: {e}", exc_info=True)
             # Use handle_error if it provides more value, otherwise just log
             # handle_error(e, "calculating semantic similarity")
             return [] # Return empty list on error

        end_time = time.time()
        logger.info(f"Semantic analysis finished. Found {len(similar_pairs)} similar pairs. Time taken: {end_time - start_time:.2f} seconds")
        return similar_pairs

# --- End of SemanticAnalyzer class ---