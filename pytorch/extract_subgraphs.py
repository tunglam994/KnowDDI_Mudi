"""
Script để extract subgraphs và lưu vào LMDB.
Có thể chạy riêng biệt, chia nhỏ theo chunks.

Ví dụ:
    # Extract tất cả splits
    python extract_subgraphs.py --dataset drugbank --splits train valid test

    # Extract chỉ train, từ index 0 đến 10000
    python extract_subgraphs.py --dataset drugbank --splits train --start_idx 0 --end_idx 10000

    # Extract train chunk tiếp theo
    python extract_subgraphs.py --dataset drugbank --splits train --start_idx 10000 --end_idx 20000
"""

import os
import argparse
import random
import numpy as np
import logging

from data_processor.subgraph_extraction import load_data, generate_subgraph_chunk

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Extract subgraphs and save to LMDB")
    
    # Dataset params
    parser.add_argument('--dataset', "-d", type=str, default='drugbank', help="Dataset name: drugbank or BioSNAP")
    parser.add_argument("--train_file", "-tf", type=str, default="train", help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid", help="Name of file containing validation triplets")
    parser.add_argument("--test_file", "-ttf", type=str, default="test", help="Name of file containing test triplets")
    parser.add_argument('--BKG_file_name', type=str, default='BKG_file', help="Background knowledge graph file name")

    # Extract subgraphs params
    parser.add_argument("--max_links", type=int, default=350000, help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=200, help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--enclosing_subgraph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False, help='whether to append adj matrix list with symmetric relations')

    # Chunk params (chia nhỏ)
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'], 
                        help="List of splits to process. E.g., --splits train valid test")
    parser.add_argument('--start_idx', type=int, default=None, help="Start index (inclusive). None = from beginning")
    parser.add_argument('--end_idx', type=int, default=None, help="End index (exclusive). None = to end")

    # Other
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    params = parser.parse_args()

    # Set seed
    np.random.seed(params.seed)
    random.seed(params.seed)
    os.environ['PYTHONHASHSEED'] = str(params.seed)

    # Set paths
    params.main_dir = os.path.dirname(os.path.realpath(__file__))
    
    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.valid_file)),
        'test': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.test_file))
    }

    params.db_path = os.path.join(params.main_dir, f'../data/{params.dataset}/digraph_hop_{params.hop}_{params.BKG_file_name}')

    print("=" * 60)
    print("EXTRACT SUBGRAPHS")
    print("=" * 60)
    print(f"Dataset: {params.dataset}")
    print(f"DB Path: {params.db_path}")
    print(f"Splits: {params.splits}")
    print(f"Start idx: {params.start_idx}")
    print(f"End idx: {params.end_idx}")
    print(f"Hop: {params.hop}")
    print(f"Max nodes per hop: {params.max_nodes_per_hop}")
    print(f"Max links: {params.max_links}")
    print("=" * 60)

    # Load data một lần
    print("\n[1/2] Loading data...")
    adj_list, files_data = load_data(params)

    # Print info về data
    print("\nData summary:")
    for split_name, split_data in files_data.items():
        print(f"  {split_name}: {len(split_data['triplets'])} triplets")

    # Extract subgraphs
    print(f"\n[2/2] Extracting subgraphs for splits: {params.splits}")
    generate_subgraph_chunk(
        params, 
        adj_list, 
        files_data, 
        target_splits=params.splits,
        start_idx=params.start_idx, 
        end_idx=params.end_idx
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
