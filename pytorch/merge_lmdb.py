"""
Script để merge nhiều LMDB databases thành một.
Dùng để ghép các chunks đã extract riêng biệt.

Ví dụ:
    # Merge db_chunk1 và db_chunk2 vào db_merged
    python merge_lmdb.py --source_dbs db_chunk1 db_chunk2 --target_db db_merged
    
    # Merge với dataset cụ thể
    python merge_lmdb.py --dataset drugbank --source_dbs ../data/drugbank/chunk1 ../data/drugbank/chunk2 --target_db ../data/drugbank/merged
"""

import os
import argparse
import lmdb
from tqdm import tqdm


def read_lmdb_split(env, split_name):
    """
    Đọc tất cả samples từ một split trong LMDB.
    
    Returns:
        dict: {index: serialized_data}
    """
    db = env.open_db(f'{split_name}_subgraph'.encode())
    data = {}
    
    with env.begin(db=db) as txn:
        # Đọc num_graphs
        num_graphs_bytes = txn.get('num_graphs'.encode())
        if num_graphs_bytes is None:
            print(f"  Warning: num_graphs not found in {split_name}")
            num_graphs = 0
        else:
            num_graphs = int.from_bytes(num_graphs_bytes, byteorder='little')
        
        print(f"  {split_name}: num_graphs = {num_graphs}")
        
        # Đọc tất cả samples
        cursor = txn.cursor(db=db)
        for key, value in cursor:
            key_str = key.decode('ascii')
            if key_str == 'num_graphs':
                continue
            
            # Parse index từ key (format: '00000123')
            try:
                idx = int(key_str)
                data[idx] = value
            except ValueError:
                print(f"  Warning: Invalid key format: {key_str}")
                continue
    
    print(f"  Loaded {len(data)} samples from {split_name}")
    return data, num_graphs


def merge_lmdb_databases(source_db_paths, target_db_path, splits=['train', 'valid', 'test']):
    """
    Merge nhiều LMDB databases thành một.
    
    Args:
        source_db_paths: List đường dẫn đến các source databases
        target_db_path: Đường dẫn target database
        splits: List các splits cần merge
    """
    print("=" * 80)
    print("MERGE LMDB DATABASES")
    print("=" * 80)
    print(f"Source DBs: {source_db_paths}")
    print(f"Target DB: {target_db_path}")
    print(f"Splits: {splits}")
    print("=" * 80)
    
    # Dict để lưu merged data: {split_name: {idx: data}}
    merged_data = {split_name: {} for split_name in splits}
    max_num_graphs = {split_name: 0 for split_name in splits}
    
    # Đọc metadata từ source đầu tiên
    metadata = {}
    
    # Đọc từng source database
    for i, source_path in enumerate(source_db_paths):
        if not os.path.isdir(source_path):
            print(f"\nWarning: Source DB not found: {source_path}")
            continue
        
        print(f"\n[{i+1}/{len(source_db_paths)}] Reading from: {source_path}")
        
        try:
            env = lmdb.open(source_path, readonly=True, max_dbs=10, lock=False)
            
            # Đọc metadata (chỉ lần đầu tiên)
            if not metadata:
                with env.begin() as txn:
                    for key in ['max_n_label_sub', 'max_n_label_obj']:
                        value = txn.get(key.encode())
                        if value is not None:
                            metadata[key] = value
            
            # Đọc từng split
            for split_name in splits:
                try:
                    split_data, num_graphs = read_lmdb_split(env, split_name)
                    
                    # Update max_num_graphs
                    max_num_graphs[split_name] = max(max_num_graphs[split_name], num_graphs)
                    
                    # Merge data: data mới ghi đè data cũ (nếu trùng index)
                    for idx, data in split_data.items():
                        if idx in merged_data[split_name]:
                            print(f"    Overwriting {split_name}[{idx}] from source {i+1}")
                        merged_data[split_name][idx] = data
                    
                except Exception as e:
                    print(f"  Error reading {split_name}: {e}")
            
            env.close()
            
        except Exception as e:
            print(f"  Error opening database: {e}")
    
    # Tính tổng kết
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    for split_name in splits:
        indices = sorted(merged_data[split_name].keys())
        if indices:
            print(f"{split_name}:")
            print(f"  Total samples: {len(indices)}")
            print(f"  Index range: [{indices[0]}:{indices[-1]+1}]")
            print(f"  Max num_graphs: {max_num_graphs[split_name]}")
            
            # Check for gaps
            expected_indices = set(range(indices[0], indices[-1] + 1))
            missing_indices = expected_indices - set(indices)
            if missing_indices:
                print(f"  Warning: Missing {len(missing_indices)} indices: {sorted(list(missing_indices))[:10]}...")
        else:
            print(f"{split_name}: No data")
    
    # Ghi vào target database
    print("\n" + "=" * 80)
    print("WRITING TO TARGET DATABASE")
    print("=" * 80)
    
    # Tính map_size (tính chính xác tổng kích thước thực tế)
    total_size = 0
    sample_count = 0
    
    print("\nCalculating total size...")
    for split_name in splits:
        for idx, data in merged_data[split_name].items():
            total_size += len(data)
            total_size += 20  # Overhead cho key và metadata
            sample_count += 1
    
    # Map size = tổng data + 50% buffer + overhead cho metadata
    map_size = int(total_size * 1.5 + 1024 * 1024 * 100)  # +100MB cho metadata
    
    print(f"Total samples: {sample_count}")
    print(f"Total data size: {total_size / (1024**3):.2f} GB")
    print(f"Map size (with buffer): {map_size / (1024**3):.2f} GB")
    
    # Tạo target directory nếu chưa có
    os.makedirs(target_db_path, exist_ok=True)
    
    env_target = lmdb.open(target_db_path, map_size=map_size, max_dbs=10)
    
    # Ghi từng split
    for split_name in splits:
        if not merged_data[split_name]:
            print(f"\nSkipping {split_name}: no data")
            continue
        
        print(f"\nWriting {split_name}...")
        db_name = f'{split_name}_subgraph'
        split_env = env_target.open_db(db_name.encode())
        
        # Ghi num_graphs
        num_graphs = max_num_graphs[split_name]
        with env_target.begin(write=True, db=split_env) as txn:
            bit_len = int.bit_length(num_graphs) if num_graphs > 0 else 1
            txn.put('num_graphs'.encode(), num_graphs.to_bytes(bit_len, byteorder='little'))
        
        # Ghi samples
        sorted_indices = sorted(merged_data[split_name].keys())
        for idx in tqdm(sorted_indices, desc=f"  Writing {split_name}"):
            key = '{:08}'.format(idx).encode('ascii')
            value = merged_data[split_name][idx]
            with env_target.begin(write=True, db=split_env) as txn:
                txn.put(key, value)
    
    # Ghi metadata
    print("\nWriting metadata...")
    with env_target.begin(write=True) as txn:
        for key, value in metadata.items():
            txn.put(key.encode(), value)
    
    env_target.close()
    
    print("\n" + "=" * 80)
    print("MERGE COMPLETED!")
    print("=" * 80)
    print(f"Target database: {target_db_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LMDB databases into one")
    
    parser.add_argument('--source_dbs', nargs='+', required=True,
                        help="List of source database paths")
    parser.add_argument('--target_db', type=str, required=True,
                        help="Target database path")
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help="List of splits to merge")
    
    args = parser.parse_args()
    
    merge_lmdb_databases(args.source_dbs, args.target_db, args.splits)


if __name__ == '__main__':
    main()
