import logging
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.sparse as ssp
import signal
from utils.data_utils import process_files_ddi, process_files_decagon
from utils.graph_utils import incidence_matrix, remove_nodes, serialize, _bfs_relational


def init_worker_signal():
    """Ignore SIGINT in worker processes để main process xử lý Ctrl+C"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def load_data(params):
    """
    Load DDI và BKG data một lần duy nhất.
    
    Returns:
        adj_list: List các adjacency matrices
        files_data: Dict chứa triplets cho mỗi split {'train': {...}, 'valid': {...}, 'test': {...}}
    """
    BKG_file = '../data/{}/{}.txt'.format(params.dataset, params.BKG_file_name)
    print(f"Loading data from BKG file: {BKG_file}")
    
    if params.dataset == 'drugbank':
        adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files_ddi(params.file_paths, BKG_file)
        triplets_mr = None
        polarity_mr = None
    elif params.dataset == 'BioSNAP':
        adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr = process_files_decagon(params.file_paths, BKG_file)

    files_data = {}
    splits = ['train', 'valid', 'test']
    
    for split_name in splits:
        if params.dataset == 'drugbank':
            split_triplets = triplets[split_name]
            split_data = {'triplets': split_triplets, 'max_size': params.max_links}
        elif params.dataset == 'BioSNAP':
            split_triplets = triplets_mr[split_name]
            split_polarity = polarity_mr[split_name]
            split_data = {'triplets': split_triplets, 'max_size': params.max_links, 'polarity_mr': split_polarity}
        
        # Áp dụng max_size
        max_size = split_data['max_size']
        original_len = len(split_data['triplets'])
        if max_size < original_len:
            perm = np.random.permutation(original_len)[:max_size]
            if isinstance(split_data['triplets'], list):
                split_data['triplets'] = [split_data['triplets'][i] for i in perm]
            else:
                split_data['triplets'] = split_data['triplets'][perm]
            if params.dataset == 'BioSNAP':
                split_data['polarity_mr'] = split_data['polarity_mr'][perm]
        
        files_data[split_name] = split_data
        print(f"  {split_name}: {len(split_data['triplets'])} triplets")
    
    return adj_list, files_data


def generate_subgraph_datasets(params):
    """
    Hàm gốc - load data và extract tất cả subgraphs.
    """
    adj_list, files_data = load_data(params)
    links2subgraphs(adj_list, files_data, params)


def generate_subgraph_chunk(params, adj_list, files_data, target_splits, start_idx=None, end_idx=None):
    """
    Extract subgraph cho một chunk cụ thể và lưu vào LMDB với đúng index.
    
    Args:
        params: Tham số cấu hình
        adj_list: Adjacency list (từ load_data)
        files_data: Dict chứa triplets (từ load_data)
        target_splits: List các split cần xử lý, VD: ['train'] hoặc ['train', 'valid', 'test']
        start_idx: Index bắt đầu (inclusive). None = 0
        end_idx: Index kết thúc (exclusive). None = cuối
    
    Ví dụ:
        adj_list, files_data = load_data(params)
        
        # Xử lý train từ 0-10000
        generate_subgraph_chunk(params, adj_list, files_data, ['train'], 0, 10000)
        
        # Xử lý train từ 10000-20000
        generate_subgraph_chunk(params, adj_list, files_data, ['train'], 10000, 20000)
        
        # Xử lý cả valid và test
        generate_subgraph_chunk(params, adj_list, files_data, ['valid', 'test'])
    """
    # Validate target_splits
    if isinstance(target_splits, str):
        target_splits = [target_splits]
    
    valid_splits = ['train', 'valid', 'test']
    for split in target_splits:
        if split not in valid_splits:
            raise ValueError(f"Split '{split}' không hợp lệ. Phải là một trong {valid_splits}")
    
    # Filter files_data theo target_splits
    filtered_data = {k: v for k, v in files_data.items() if k in target_splits}
    
    print(f"Processing splits: {target_splits}, start_idx: {start_idx}, end_idx: {end_idx}")
    
    links2subgraphs_chunk(adj_list, filtered_data, params, start_idx=start_idx, end_idx=end_idx)


def links2subgraphs(adj_list, files_data, params, max_label_value=None):
    '''
    Extract enclosing subgraphs (hàm gốc - xử lý tất cả).
    '''
    links2subgraphs_chunk(adj_list, files_data, params, max_label_value=max_label_value, 
                          start_idx=None, end_idx=None)


def links2subgraphs_chunk(adj_list, files_data, params, max_label_value=None, start_idx=None, end_idx=None):
    '''
    Extract enclosing subgraphs cho một chunk và lưu vào LMDB với đúng index.
    
    Args:
        start_idx: Index bắt đầu (inclusive). None = 0
        end_idx: Index kết thúc (exclusive). None = cuối
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    # Tính map_size cho LMDB
    first_split_triplets = list(files_data.values())[0]['triplets']
    sample_size = min(100, len(first_split_triplets))
    BYTES_PER_DATUM = get_average_subgraph_size(sample_size, first_split_triplets, adj_list, params) * 2.0
    
    links_length = 0
    for split_name, split in files_data.items():
        links_length += len(split['triplets']) * 2
    map_size = int(links_length * BYTES_PER_DATUM)

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(adj_list, links, g_labels, split_env, actual_start, actual_end, total_count):
        """
        Extract subgraphs và lưu với đúng index.
        
        Args:
            links: Triplets đã slice
            g_labels: Labels đã slice
            actual_start: Index bắt đầu thực tế trong LMDB
            actual_end: Index kết thúc thực tế
            total_count: Tổng số samples trong split (để ghi num_graphs)
        """
        # Ghi num_graphs chỉ khi start_idx = 0 (lần đầu tiên)
        if actual_start == 0:
            with env.begin(write=True, db=split_env) as txn:
                bit_len = int.bit_length(total_count) if total_count > 0 else 1
                txn.put('num_graphs'.encode(), total_count.to_bytes(bit_len, byteorder='little'))
        
        pool = None
        try:
            pool = mp.Pool(processes=4, initializer=intialize_worker, initargs=(adj_list, params, max_label_value))
            # Index thực tế = actual_start + local_idx
            args_ = zip(
                range(actual_start, actual_end),  # Index thực tế trong LMDB
                links,
                g_labels
            )
            desc = f"Extracting [{actual_start}:{actual_end}]"
            for (str_id, datum) in tqdm(pool.imap(extract_save_subgraph, args_), total=len(links), desc=desc):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])
                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user! Terminating workers...")
            pool.terminate()
            pool.join()
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    for split_name, split in files_data.items():
        all_triplets = split['triplets']
        total_count = len(all_triplets)
        
        # Xác định range
        actual_start = start_idx if start_idx is not None else 0
        actual_end = end_idx if end_idx is not None else total_count
        
        # Clamp range
        actual_start = max(0, min(actual_start, total_count))
        actual_end = max(actual_start, min(actual_end, total_count))
        
        if actual_start >= actual_end:
            print(f"Skipping {split_name}: no data in range [{actual_start}:{actual_end}]")
            continue
        
        logging.info(f"Extracting subgraphs for {split_name}: [{actual_start}:{actual_end}] / {total_count}")
        print(f"Processing {split_name}: [{actual_start}:{actual_end}] of {total_count} samples")
        
        # Slice triplets
        if isinstance(all_triplets, list):
            slice_triplets = all_triplets[actual_start:actual_end]
        else:
            slice_triplets = all_triplets[actual_start:actual_end]
        
        # Slice labels
        if params.dataset == 'BioSNAP':
            all_polarity = split['polarity_mr']
            slice_polarity = all_polarity[actual_start:actual_end]
            g_labels = np.array(slice_polarity)
        else:
            g_labels = np.ones(len(slice_triplets))
        
        db_name_pos = split_name + '_subgraph'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(adj_list, slice_triplets, g_labels, split_env, 
                         actual_start, actual_end, total_count)

    # Ghi metadata
    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']
    with env.begin(write=True) as txn:
        bit_len_sub = int.bit_length(int(max_n_label['value'][0])) if max_n_label['value'][0] > 0 else 1
        bit_len_obj = int.bit_length(int(max_n_label['value'][1])) if max_n_label['value'][1] > 0 else 1
        txn.put('max_n_label_sub'.encode(), int(max_n_label['value'][0]).to_bytes(bit_len_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), int(max_n_label['value'][1]).to_bytes(bit_len_obj, byteorder='little'))



def get_average_subgraph_size(sample_size, links, adj_list, params):
    total_size = 0
    lst = np.random.choice(len(links), sample_size)
    for idx in lst:
        (n1, n2, r_label) = links[idx]
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, adj_list, params.hop, params.enclosing_subgraph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(adj_list, params, max_label_value):
    global adj_list_, params_, max_label_value_
    adj_list_, params_, max_label_value_ = adj_list, params, max_label_value


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    pruned_subgraph_nodes, pruned_node_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, adj_list_, params_.hop, params_.enclosing_subgraph, params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        pruned_node_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in pruned_node_labels])

    datum = {'nodes': pruned_subgraph_nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': pruned_node_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes

def get_neighbor_nodes(roots, adj, hop=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(hop):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)

def subgraph_extraction_labeling(ind, rel, A_list, hop=1, enclosing_subgraph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T
    ind = list(ind)
    ind[0], ind[1] = int(ind[0]), int(ind[1])
    ind = (ind[0], ind[1])
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, hop, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, hop, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_subgraph:
        if ind[0] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_int:
            subgraph_nei_nodes_int.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        if ind[0] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[0])
        if ind[1] in subgraph_nei_nodes_un:
            subgraph_nei_nodes_un.remove(ind[1])
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
    
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=hop)
    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes



