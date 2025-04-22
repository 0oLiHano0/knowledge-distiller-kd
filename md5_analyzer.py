import hashlib
import collections
from pathlib import Path # Path might be part of the block_info tuple

# Type hinting for clarity
from typing import List, Tuple, Dict, Any

# Define the structure of block_info for type hinting
# (file_path: Path, block_index: int, block_type: str, block_text: str)
BlockInfo = Tuple[Path, int, str, str]
DuplicateGroups = Dict[str, List[BlockInfo]]

def find_md5_duplicates(blocks_data: List[BlockInfo]) -> DuplicateGroups:
    """
    Calculates MD5 hashes for text blocks and identifies duplicates.

    Args:
        blocks_data: A list where each element is a tuple:
                     (file_path: Path, block_index: int, block_type: str, block_text: str).

    Returns:
        A dictionary where keys are MD5 hashes and values are lists of
        block_info tuples that share that hash (only includes groups with > 1 block).
    """
    print(f"  [MD5 Analyzer] Received {len(blocks_data)} blocks for analysis.")
    hash_map = collections.defaultdict(list)
    processed_count = 0

    for block_info in blocks_data:
        # Ensure block_info has the expected structure
        if len(block_info) != 4:
             print(f"  [MD5 Analyzer] [Warning] Skipping invalid block_info format: {block_info}")
             continue

        file_path, b_index, b_type, b_text = block_info

        # Ensure b_text is a string before encoding
        if not isinstance(b_text, str):
            print(f"  [MD5 Analyzer] [Warning] Block text is not a string for block {b_index} in {file_path.name}. Skipping.")
            continue

        try:
            # Calculate MD5 hash of the block text
            # Encode the text to bytes (UTF-8 is standard) before hashing
            hash_object = hashlib.md5(b_text.encode('utf-8'))
            hex_dig = hash_object.hexdigest()

            # Append the full block info to the list associated with this hash
            hash_map[hex_dig].append(block_info)
            processed_count += 1
        except Exception as e:
            # Catch potential errors during encoding or hashing
            print(f"  [MD5 Analyzer] [Error] Failed to process block {b_index} in {file_path.name}: {e}")

    print(f"  [MD5 Analyzer] Processed {processed_count} blocks.")

    # Filter the hash_map to include only hashes with more than one associated block
    duplicate_blocks = {h: b_list for h, b_list in hash_map.items() if len(b_list) > 1}

    print(f"  [MD5 Analyzer] Found {len(duplicate_blocks)} groups of MD5 duplicates.")
    return duplicate_blocks

# Example Usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Create some dummy data
    dummy_path1 = Path("./dummy_file1.md")
    dummy_path2 = Path("./dummy_file2.md")
    dummy_blocks = [
        (dummy_path1, 0, 'paragraph', 'This is the first block.'),
        (dummy_path1, 1, 'block_code', 'print("Hello")'),
        (dummy_path2, 0, 'paragraph', 'This is the first block.'), # Duplicate of block 0
        (dummy_path1, 2, 'paragraph', 'This is a unique block.'),
        (dummy_path2, 1, 'block_code', 'print("Hello")'), # Duplicate of block 1
        (dummy_path2, 2, 'list_item', 'Item A'),
    ]

    print("Running MD5 Analyzer example...")
    duplicates = find_md5_duplicates(dummy_blocks)

    if duplicates:
        print("\nDuplicate Groups Found:")
        for i, (hash_val, group) in enumerate(duplicates.items()):
            print(f"\nGroup {i+1} (Hash: {hash_val}):")
            for item in group:
                print(f"  - File: {item[0].name}, Index: {item[1]}, Type: {item[2]}, Preview: '{item[3][:30]}...'")
    else:
        print("\nNo duplicate groups found.")
