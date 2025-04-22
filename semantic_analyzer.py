import time
import hashlib
from sentence_transformers import SentenceTransformer, util
from pathlib import Path # Path might be part of the block_info tuple

# Type hinting
from typing import List, Tuple, Optional, Any

# Define the structure of block_info for type hinting
# (file_path: Path, block_index: int, block_type: str, block_text: str)
BlockInfo = Tuple[Path, int, str, str]
SemanticPair = Tuple[BlockInfo, BlockInfo, float] # (block1_info, block2_info, similarity_score)

class SemanticAnalyzer:
    """
    Handles semantic similarity analysis using Sentence Transformers.
    Loads the model upon initialization.
    """
    # Default model - consider making this configurable
    DEFAULT_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

    def __init__(self, model_name: Optional[str] = None):
        """
        Initializes the analyzer and loads the Sentence Transformer model.

        Args:
            model_name (Optional[str]): The name of the Sentence Transformer model to use.
                                        If None, uses DEFAULT_MODEL_NAME.
        Raises:
            ImportError: If sentence_transformers is not installed.
            Exception: If the model fails to load for other reasons.
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self):
        """Loads the sentence transformer model."""
        print(f"  [Semantic Analyzer] Attempting to load model: {self.model_name}...")
        start_time = time.time()
        try:
            # This might download the model on first run
            self.model = SentenceTransformer(self.model_name)
            end_time = time.time()
            print(f"  [Semantic Analyzer] Model '{self.model_name}' loaded successfully in {end_time - start_time:.2f} seconds.")
        except ImportError:
             print(f"[严重错误] [Semantic Analyzer] 'sentence-transformers' library not found.")
             print("Please install it: pip install sentence-transformers")
             # Re-raise to signal failure clearly to the main application
             raise
        except Exception as e:
            print(f"[严重错误] [Semantic Analyzer] Failed to load model '{self.model_name}': {e}")
            self.model = None # Ensure model is None if loading failed
            # Re-raise the exception so the main app knows initialization failed
            raise Exception(f"Failed to load semantic model '{self.model_name}': {e}") from e

    def is_model_loaded(self) -> bool:
        """Checks if the model was loaded successfully."""
        return self.model is not None

    def _calculate_embeddings(self, texts: List[str]) -> Optional[Any]:
        """Calculates embeddings for a list of texts."""
        if not self.is_model_loaded() or not texts:
            print("  [Semantic Analyzer] Model not loaded or no texts provided for embedding calculation.")
            return None
        try:
            print(f"  [Semantic Analyzer] Calculating embeddings for {len(texts)} texts...")
            start_time = time.time()
            # Use the loaded model to encode texts
            # convert_to_tensor=True is generally good for performance with util.semantic_search
            embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False) # Disable progress bar for cleaner logs
            end_time = time.time()
            print(f"  [Semantic Analyzer] Embedding calculation finished in {end_time - start_time:.2f} seconds.")
            return embeddings
        except Exception as e:
            print(f"[错误] [Semantic Analyzer] Error during embedding calculation: {e}")
            return None

    def find_similar_blocks(self,
                            blocks_info: List[BlockInfo],
                            threshold: float = 0.85,
                            skip_md5_duplicates: bool = True) -> List[SemanticPair]:
        """
        Finds pairs of blocks with semantic similarity above a given threshold.

        Args:
            blocks_info: A list of block information tuples.
            threshold: The minimum cosine similarity score to consider a pair as similar.
            skip_md5_duplicates: If True, skips comparing pairs that have identical MD5 hashes.

        Returns:
            A list of tuples, where each tuple contains:
            (block_info1, block_info2, similarity_score).
            Returns an empty list if the model is not loaded or an error occurs.
        """
        similar_pairs: List[SemanticPair] = []
        if not self.is_model_loaded():
            print("  [Semantic Analyzer] Model not loaded. Cannot find similar blocks.")
            return similar_pairs
        if len(blocks_info) < 2:
            print("  [Semantic Analyzer] Less than 2 blocks provided. No pairs to compare.")
            return similar_pairs

        # 1. Extract texts for embedding calculation
        block_texts = [info[3] for info in blocks_info if len(info) == 4 and isinstance(info[3], str)]
        if len(block_texts) != len(blocks_info):
             print(f"  [Semantic Analyzer] [Warning] Mismatch between blocks_info ({len(blocks_info)}) and valid texts extracted ({len(block_texts)}). Check block_info format.")
             # Filter blocks_info to match the texts we are actually embedding
             valid_blocks_info = [info for info in blocks_info if len(info) == 4 and isinstance(info[3], str)]
        else:
             valid_blocks_info = blocks_info


        if len(valid_blocks_info) < 2:
             print("  [Semantic Analyzer] Less than 2 valid blocks after filtering. No pairs to compare.")
             return similar_pairs


        # 2. Calculate Embeddings
        embeddings = self._calculate_embeddings(block_texts)
        if embeddings is None:
            print("  [Semantic Analyzer] Failed to calculate embeddings. Cannot find similar blocks.")
            return similar_pairs # Return empty list on embedding failure

        # 3. Perform Semantic Search
        print(f"  [Semantic Analyzer] Performing semantic search with threshold > {threshold}...")
        start_time = time.time()
        try:
            # util.semantic_search finds the most similar items in the corpus for each query item.
            # Here, we search the embeddings against themselves.
            # top_k=len(embeddings) ensures we get all potential pairs above the threshold (we filter later)
            hits = util.semantic_search(
                embeddings, embeddings,
                query_chunk_size=100,  # Adjust chunk sizes based on memory/performance
                corpus_chunk_size=500,
                top_k=len(valid_blocks_info),         # Get enough candidates initially
                score_function=util.cos_sim # Standard cosine similarity
            )
            search_time = time.time()
            print(f"  [Semantic Analyzer] Initial search completed in {search_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"[错误] [Semantic Analyzer] Error during semantic search: {e}")
            return similar_pairs # Return empty list on search failure

        # 4. Process Hits and Filter Pairs
        processed_pairs = set() # To avoid adding (A, B) and (B, A)
        md5_skipped_count = 0

        # Calculate MD5 hashes beforehand if skipping MD5 duplicates
        block_hashes = {}
        if skip_md5_duplicates:
            print("  [Semantic Analyzer] Calculating MD5 hashes to skip exact duplicates...")
            try:
                 block_hashes = {
                     i: hashlib.md5(valid_blocks_info[i][3].encode('utf-8')).hexdigest()
                     for i in range(len(valid_blocks_info))
                 }
            except Exception as e:
                 print(f"  [Semantic Analyzer] [Warning] Error calculating MD5 hashes: {e}. Will not skip exact duplicates.")
                 skip_md5_duplicates = False # Disable skipping if hashing fails

        print("  [Semantic Analyzer] Filtering search results...")
        filter_start_time = time.time()
        # Each `hits[i]` contains a list of {'corpus_id': j, 'score': s} for query i
        for query_idx in range(len(hits)):
            query_block_info = valid_blocks_info[query_idx]
            query_hash = block_hashes.get(query_idx) if skip_md5_duplicates else None

            for hit in hits[query_idx]:
                corpus_idx = hit['corpus_id']
                score = hit['score']

                # Skip self-comparison
                if query_idx == corpus_idx:
                    continue

                # Skip if score is below threshold
                if score < threshold:
                    continue # Scores are usually sorted, so we might break early, but util might not guarantee order

                # Skip if this pair (in any order) has already been processed
                pair_indices = tuple(sorted((query_idx, corpus_idx)))
                if pair_indices in processed_pairs:
                    continue

                # Optional: Skip if blocks have identical MD5 hashes
                if skip_md5_duplicates and query_hash is not None:
                    corpus_hash = block_hashes.get(corpus_idx)
                    if corpus_hash is not None and query_hash == corpus_hash:
                        md5_skipped_count += 1
                        processed_pairs.add(pair_indices) # Mark as processed even if skipped
                        continue # Skip this pair

                # If all checks pass, add the pair
                corpus_block_info = valid_blocks_info[corpus_idx]
                similar_pairs.append((query_block_info, corpus_block_info, score))
                processed_pairs.add(pair_indices) # Mark this pair as processed

        filter_end_time = time.time()
        if skip_md5_duplicates:
            # The count might be double if (A,B) and (B,A) were both skipped, divide by 2 for unique pairs skipped.
            print(f"  [Semantic Analyzer] Skipped {md5_skipped_count // 2} pairs due to identical MD5 hash.")
        print(f"  [Semantic Analyzer] Filtering completed in {filter_end_time - filter_start_time:.2f} seconds.")
        print(f"  [Semantic Analyzer] Found {len(similar_pairs)} semantic pairs above threshold {threshold}.")

        return similar_pairs

# Example Usage (optional, for testing the module directly)
if __name__ == '__main__':
    print("Running Semantic Analyzer example...")
    try:
        analyzer = SemanticAnalyzer() # Load default model

        if analyzer.is_model_loaded():
            # Create some dummy data
            dummy_path1 = Path("./dummy_file1.md")
            dummy_path2 = Path("./dummy_file2.md")
            dummy_blocks = [
                (dummy_path1, 0, 'paragraph', 'The quick brown fox jumps over the lazy dog.'),
                (dummy_path1, 1, 'block_code', 'def func(): pass'),
                (dummy_path2, 0, 'paragraph', 'A fast, dark-colored fox leaps above a sleepy canine.'), # Similar to block 0
                (dummy_path1, 2, 'paragraph', 'This is a completely different sentence about weather.'),
                (dummy_path2, 1, 'block_code', 'def func():\n    pass'), # Slightly different whitespace, but semantically same code?
                (dummy_path2, 2, 'list_item', 'Apple'),
                (dummy_path1, 3, 'paragraph', 'The speedy brown fox hurdles the resting hound.'), # Similar to block 0 & 2
            ]

            similarity_threshold = 0.7 # Lower threshold for example

            print(f"\nFinding pairs with similarity > {similarity_threshold}...")
            found_pairs = analyzer.find_similar_blocks(dummy_blocks, threshold=similarity_threshold, skip_md5_duplicates=True)

            if found_pairs:
                print(f"\nFound {len(found_pairs)} similar pairs:")
                # Sort pairs by score descending for better readability
                found_pairs.sort(key=lambda x: x[2], reverse=True)
                for i, (info1, info2, score) in enumerate(found_pairs):
                    print(f"\nPair {i+1} (Score: {score:.4f}):")
                    print(f"  - Block 1: {info1[0].name} ({info1[2]} #{info1[1]}) '{info1[3][:50]}...'")
                    print(f"  - Block 2: {info2[0].name} ({info2[2]} #{info2[1]}) '{info2[3][:50]}...'")
            else:
                print("\nNo similar pairs found above the threshold.")

        else:
             print("\nSemantic model failed to load. Cannot run example.")

    except Exception as e:
        print(f"\nAn error occurred during the example: {e}")

