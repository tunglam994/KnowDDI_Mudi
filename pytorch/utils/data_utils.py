import numpy as np
from scipy.sparse import csc_matrix

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

def process_files_ddi(files, BKG_file, keeptrainone=False):
    entity2id = {}
    relation2id = {}

    # Hai bộ triplets: full (dùng subgraph, giữ cả + và -) và adj-only (chỉ dương)
    triplets_full = {}
    triplets_for_adj = {}

    kg_triple = []  # will store mapped ids [h_id, t_id, r_rel]
    rel = 0   # số relation dương trong DDI

    # -------------------------
    # 1) đọc files DDI
    # -------------------------
    for file_type, file_path in files.items():
        full_list = []   # (h_original, t_original, r) hoặc mapped? chúng ta map entities ngay
        adj_list = []    # chỉ dương, dùng để build adj

        file_data = np.loadtxt(file_path)
        for triplet in file_data:
            h_raw, t_raw, r_raw = int(triplet[0]), int(triplet[1]), int(triplet[2])

            # map entity -> dense id 0..N-1
            if h_raw not in entity2id:
                entity2id[h_raw] = len(entity2id)
            if t_raw not in entity2id:
                entity2id[t_raw] = len(entity2id)
            h = entity2id[h_raw]
            t = entity2id[t_raw]
            r = int(r_raw)

            # Lưu full (mapped entity ids, relation as-is; negative kept)
            full_list.append([h, t, r])

            # Nếu relation dương thì thêm vào adj-only
            if r != -1:
                if r not in relation2id:
                    # map relation -> some id (ở đây ta giữ nguyên label r; shift KG relations later)
                    relation2id[r] = r
                    rel += 1
                adj_list.append([h, t, r])

        # chuyển sang numpy int32 để tiết kiệm RAM
        triplets_full[file_type] = np.array(full_list, dtype=np.int32)
        triplets_for_adj[file_type] = np.array(adj_list, dtype=np.int32)

    # -------------------------
    # 2) đọc KG, map entity ids và relation ids (shift by rel)
    # -------------------------
    triplet_kg_raw = np.loadtxt(BKG_file)
    kg_rows = []
    kg_cols = []
    kg_rels = []
    for (h_raw, t_raw, r_raw) in triplet_kg_raw:
        h_raw, t_raw, r_raw = int(h_raw), int(t_raw), int(r_raw)

        if h_raw not in entity2id:
            entity2id[h_raw] = len(entity2id)
        if t_raw not in entity2id:
            entity2id[t_raw] = len(entity2id)
        h = entity2id[h_raw]
        t = entity2id[t_raw]
        r = int(r_raw)

        new_r = rel + r
        if new_r not in relation2id:
            relation2id[new_r] = new_r

        kg_rows.append(h)
        kg_cols.append(t)
        kg_rels.append(r)  # store original r relative to KG; we'll use (i - rel) when grouping

    # make kg_triple as numpy int32 Nx3 with mapped entity indices and original kg relation index in col2
    kg_triple = np.vstack([np.array(kg_rows, dtype=np.int32),
                            np.array(kg_cols, dtype=np.int32),
                            np.array(kg_rels, dtype=np.int32)]).T

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # -------------------------
    # 3) Build adjacency list using triplets_for_adj (mapped ids) and kg_triple (mapped ids)
    # -------------------------
    n_entities = len(entity2id)
    adj_list = []

    # DDI relations (0..rel-1)
    for r_pos in range(rel):
        # select rows where relation equals r_pos
        arr = triplets_for_adj['train']
        if arr.size == 0:
            # no positive edges for this split/relation
            adj_list.append(csc_matrix((n_entities, n_entities), dtype=np.uint8))
            continue

        mask_idx = np.where(arr[:, 2] == r_pos)[0]
        if mask_idx.size == 0:
            adj_list.append(csc_matrix((n_entities, n_entities), dtype=np.uint8))
            continue

        rows = arr[mask_idx, 0].astype(np.int32)
        cols = arr[mask_idx, 1].astype(np.int32)
        data = np.ones(rows.shape[0], dtype=np.uint8)
        coo = coo_matrix((data, (rows, cols)), shape=(n_entities, n_entities))
        adj_list.append(coo.tocsc())

    # KG relations: relation ids in relation2id are shifted by rel
    kg_rels_np = kg_triple[:, 2].astype(np.int32)
    for i in range(rel, len(relation2id)):
        kg_rel_idx = i - rel
        mask_idx = np.where(kg_rels_np == kg_rel_idx)[0]
        if mask_idx.size == 0:
            adj_list.append(csc_matrix((n_entities, n_entities), dtype=np.uint8))
            continue
        rows = kg_triple[mask_idx, 0].astype(np.int32)
        cols = kg_triple[mask_idx, 1].astype(np.int32)
        data = np.ones(rows.shape[0], dtype=np.uint8)
        coo = coo_matrix((data, (rows, cols)), shape=(n_entities, n_entities))
        adj_list.append(coo.tocsc())

    return adj_list, triplets_full, entity2id, relation2id, id2entity, id2relation, rel


def process_files_decagon(files, triple_file, keeptrainone = False):
    entity2id = {}
    relation2id = {}

    triplets = {}
    triplets_mr = {}
    polarity_mr = {}
    kg_triple = []
    triplets_train = []
    rel = 0

    for file_type, file_path in files.items():
        data = []
        data_mr = []
        data_pol = []
        with open(file_path, 'r') as f:
            for lines in f:
                h, t, r, p = lines.strip().split('\t')
                h, t = int(h), int(t)
                p = int(p) # pos/neg edge
                list_r_onehot = list(map(int, r.split(',')))
                list_r = [0] if keeptrainone else [i for i, _ in enumerate(list_r_onehot) if _ == 1]  
                for s in list_r:
                    triplet = [h,t,s]
                    triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    if triplet[0] not in entity2id:
                        entity2id[triplet[0]] = triplet[0]
                    if triplet[1] not in entity2id:
                        entity2id[triplet[1]] = triplet[1]
                    if triplet[2] not in relation2id:
                        if keeptrainone:
                            triplet[2] = 0
                            relation2id[triplet[2]] = 0
                            rel = 1
                        else:
                            relation2id[triplet[2]] = triplet[2]
                            rel += 1
                    # Save the triplets corresponding to only the known relations
                    if triplet[2] in relation2id :
                        data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                        if file_type == 'train' and p == 1:
                            triplets_train.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                if keeptrainone:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], 0])
                else:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], list_r_onehot])
                data_pol.append(p)
        triplets_train = np.array(triplets_train)
        triplets[file_type] = np.array(data)#merged triplets (h,r,t)
        triplets_mr[file_type] = data_mr# triplets (h,r,[t1,t2,t3,....,tn]) ti=0/1
        polarity_mr[file_type] = np.array(data_pol)#whether the fake triplets

    assert len(entity2id) == 604
    if not keeptrainone:
        assert rel == 200
    else:
        assert rel == 1
    triplet_kg = np.loadtxt(triple_file)
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        # same id within train/valid/test and BKG_file does not mean same relation
        if rel+r not in relation2id:
            relation2id[rel+r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed from the train data and BKG data.
    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets_train[:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets_train[:, 0][idx].squeeze(1), triplets_train[:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr