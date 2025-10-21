import csv
import random
import time
import json
import os
import shutil
from shutil import copyfile

import torch
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as ssp
from tqdm import tqdm

RDF_type_string = 'rdf:type'


# -------------------------------------------------------------------------
# Normalize dataset items to binary (h, r, t, label) tuples
# -------------------------------------------------------------------------
def normalize_to_binary(dataset, is_test=False):
    """
    Accepts mixed records and returns a list of (h, r, t, label:str) tuples.

    Supported input shapes per item:
      - (h, r, t)                      -> label = '1'
      - (h, r, t, label)
      - (rel, [e1, e2, ...])          -> expanded to (e1, rel, ei), label = '1'
      - (rel, [e1, e2, ...], label)   -> expanded to (e1, rel, ei) with the same label

    Note:
      - We ignore rdf:type when building edges, but keep it for features/masks later.
      - Label is always returned as string '1' or '0' for compatibility with legacy code.
    """
    out = []
    for item in dataset:
        # Multi-ary form: (rel, [e1, e2, ...]) or (rel, [e1, e2, ...], label)
        if len(item) >= 2 and isinstance(item[1], list):
            rel = item[0]
            ents = item[1]
            if len(item) >= 3:
                lab = item[2]
                lab = '1' if (str(lab).lower() in ('1', 'true')) else '0'
            else:
                lab = '1'
            if len(ents) >= 2:
                head = ents[0]
                for i in range(1, len(ents)):
                    tail = ents[i]
                    out.append((head, rel, tail, lab))
            # if only one entity, skip as it cannot form a binary edge
            continue

        # Triplet forms
        if is_test:
            # In test mode we often pass 3-tuples (h,r,t)
            if len(item) == 3:
                h, r, t = item
                out.append((h, r, t, '1'))
            elif len(item) >= 4:
                h, r, t, lab = item[:4]
                lab = '1' if (str(lab).lower() in ('1', 'true')) else '0'
                out.append((h, r, t, lab))
        else:
            if len(item) == 3:
                h, r, t = item
                out.append((h, r, t, '1'))
            elif len(item) >= 4:
                h, r, t, lab = item[:4]
                lab = '1' if (str(lab).lower() in ('1', 'true')) else '0'
                out.append((h, r, t, lab))
    return out


# -------------------------------------------------------------------------
# Helper: compute 2-hop entity pairs
# -------------------------------------------------------------------------
def get_2_hop_pairs(dataset, is_test=False):
    """Compute all 2-hop reachable entity pairs from dataset (after normalization)."""
    flat = normalize_to_binary(dataset, is_test=is_test)

    edges, ent2id, rel2id = [], {}, {}
    entities, relations = [], []
    num_ent, num_rel = 0, 0

    for (h, r, t, lab) in flat:
        if lab == '0' or r == RDF_type_string:
            continue
        for ent in (h, t):
            if ent not in ent2id:
                ent2id[ent] = num_ent
                num_ent += 1
                entities.append(ent)
        if r not in rel2id:
            rel2id[r] = num_rel
            num_rel += 1
            relations.append(r)
        edges.append((ent2id[h], rel2id[r], ent2id[t]))

    if len(edges) == 0 or num_ent == 0:
        return [], ssp.coo_matrix((0, 0)), ent2id, entities

    row = np.array([e[0] for e in edges])
    col = np.array([e[2] for e in edges])
    adj_matrix = ssp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_ent, num_ent))
    adj_2hop = dict((adj_matrix @ adj_matrix).todok())

    new_pairs = [(entities[i], entities[j]) for (i, j) in adj_2hop.keys() if i != j]
    return new_pairs, adj_matrix, ent2id, entities


# -------------------------------------------------------------------------
# Load predicate lists (binary/unary)
# -------------------------------------------------------------------------
def load_predicates(dataset_name):
    """
    Load binary and unary predicates (relations) from dataset folder.
    Works for FB-AUTO style datasets that have relations.txt / entities.txt.

    Returns:
        binaryPredicates: list of relation names
        unaryPredicates:  [] (empty, FB-AUTO没有单元谓词)
    """
    base_path = f"data/{dataset_name}"
    rel_path_txt = os.path.join(base_path, "relations.txt")
    rel_path_dict = os.path.join(base_path, "relations.dict")

    relations = []
    if os.path.exists(rel_path_txt):
        with open(rel_path_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    _, rel = parts
                else:
                    rel = parts[0]
                relations.append(rel)
    elif os.path.exists(rel_path_dict):
        with open(rel_path_dict, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    _, rel = parts
                else:
                    rel = parts[0]
                relations.append(rel)
    else:
        raise FileNotFoundError(f"No relations file found in {base_path}")

    binaryPredicates = relations
    unaryPredicates = []
    print(f"Loaded {len(binaryPredicates)} relations for {dataset_name}")
    return binaryPredicates, unaryPredicates


# -------------------------------------------------------------------------
# Graph encoding
# -------------------------------------------------------------------------
def encode_input_dataset(input_dataset, query_dataset, binaryPredicates, unaryPredicates,
                         add_2hop=True, valid_examples=None, is_test=False):
    """
    Convert input + query triples into a GNN-compatible graph (node/edge/mask).
    Handles unary/binary predicates and approximates higher-arity facts via pair nodes.
    """
    start_time = time.time()
    num_binary = len(binaryPredicates)
    num_unary = len(unaryPredicates)
    feature_dimension = num_binary + num_unary

    # predicate → column index
    pred_dict = {p: i for i, p in enumerate(binaryPredicates)}
    for i, p in enumerate(unaryPredicates):
        pred_dict[p] = num_binary + i

    all_constants, all_pairs = set(), set()

    # ✅ 强制收集所有训练 triple 的 (h, t)
    for data in input_dataset + query_dataset:
        if is_test:
            h, r, t = data
        else:
            h, r, t, label = data
        if r != RDF_type_string:
            all_pairs.add((h, t))
            all_constants.update([h, t])
        else:
            all_constants.add(h)

    # ✅ 如果配置启用 2-hop，也扩展邻居
    if add_2hop:
        new_pairs, _, _, _ = get_2_hop_pairs(input_dataset, is_test)
        all_pairs.update(new_pairs)

    singleton_nodes = list(all_constants)
    num_singleton_nodes = len(singleton_nodes)
    pair_nodes = list(all_pairs)

    const_node_dict = {const: i for i, const in enumerate(singleton_nodes)}
    node_to_const_dict = {i: c for i, c in enumerate(singleton_nodes)}

    # ✅ 将 pair 节点加入 const_node_dict
    for i, pair in enumerate(pair_nodes):
        const_node_dict[pair] = num_singleton_nodes + i
        node_to_const_dict[num_singleton_nodes + i] = pair

    # 构造边
    edge_list, edge_type_list = [], []
    for i, pair in enumerate(pair_nodes):
        pid = num_singleton_nodes + i
        if pair[0] in const_node_dict and pair[1] in const_node_dict:
            edge_list += [
                (const_node_dict[pair[0]], pid),
                (pid, const_node_dict[pair[0]]),
                (const_node_dict[pair[1]], pid),
                (pid, const_node_dict[pair[1]]),
            ]
            edge_type_list += [0, 0, 1, 1]

    edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.LongTensor([[], []])
    edge_types = torch.LongTensor(edge_type_list)

    # 构造节点特征
    x = np.zeros((len(singleton_nodes) + len(pair_nodes), feature_dimension))
    for item in input_dataset:
        if is_test:
            h, r, t = item
            label = '1'
        else:
            h, r, t, label = item
        if r == RDF_type_string:
            idx_node = const_node_dict[h]
            idx_pred = pred_dict.get(t, None)
        else:
            idx_node = const_node_dict.get((h, t), None)
            idx_pred = pred_dict.get(r, None)
        if idx_node is not None and idx_pred is not None and label == '1':
            x[idx_node][idx_pred] = 1

    x = torch.FloatTensor(x)

    print(f"[DEBUG] Graph built: {len(singleton_nodes)} singletons, {len(pair_nodes)} pairs, "
          f"x.shape={x.shape}, preds={len(pred_dict)}")

    return x, edge_index, edge_types, node_to_const_dict, const_node_dict, pred_dict, num_singleton_nodes



# -------------------------------------------------------------------------
# Process dataset for constants/pairs
# -------------------------------------------------------------------------
def process(input_dataset, query_dataset, add_2hop, valid_examples=None, is_test=False):
    all_constants, all_pairs = set(), set()

    flat_input = normalize_to_binary(input_dataset, is_test=False)
    flat_query = normalize_to_binary(query_dataset, is_test=is_test)

    for (h, r, t, _) in flat_input + flat_query:
        if r == RDF_type_string:
            all_constants.add(h)
        else:
            all_pairs.add((h, t))
            all_constants.update([h, t])

    if add_2hop:
        new_pairs, _, _, _ = get_2_hop_pairs(input_dataset, is_test=False)
        for p in new_pairs:
            all_pairs.add(p)

    if valid_examples:
        for buf in valid_examples:
            parts = buf.strip().split()
            if len(parts) >= 3:
                h, r, t = parts[:3]
                if r == RDF_type_string:
                    all_constants.add(h)
                else:
                    all_pairs.add((h, t))
                    all_constants.update([h, t])

    return all_constants, all_pairs


# -------------------------------------------------------------------------
# Label + mask construction
# -------------------------------------------------------------------------
# def generate_labels_and_mask(dataset, node_to_const_dict, const_to_node_dict, pred_dict):
#     print("\n=== DEBUG: Checking label mapping ===")
#     print(f"Sample from dataset: {dataset[0]}")
#     print(f"Pred_dict keys (first 10): {list(pred_dict.keys())[:10]}")
#
#     num_nodes, num_preds = len(node_to_const_dict), len(pred_dict)
#     labels = np.zeros((num_nodes, num_preds), dtype=np.float32)
#     mask = np.zeros((num_nodes, num_preds), dtype=np.float32)
#
#     flat = normalize_to_binary(dataset, is_test=False)
#
#     for (h, r, t, lab) in flat:
#         if r == RDF_type_string:
#             idx_node = const_to_node_dict.get(h, None)
#             idx_pred = pred_dict.get(t, None)
#         else:
#             # ⚙️ FB-AUTO 是 (relation, head, tail)，要反过来取
#             idx_node = const_to_node_dict.get((t, h), None)
#             if idx_node is None:
#                 idx_node = const_to_node_dict.get((h, t), None)
#             idx_pred = pred_dict.get(r, None)
#
#         if idx_node is not None and idx_pred is not None:
#             mask[idx_node, idx_pred] = 1.0
#
#             # --- FIX: robust label parsing & keep-1 policy ---
#             if isinstance(lab, (int, float)):
#                 new_label = 1.0 if float(lab) > 0.5 else 0.0
#             else:
#                 lab_str = str(lab).strip().lower()
#                 new_label = 1.0 if lab_str in ('1', 'true', 'yes') else 0.0
#
#             labels[idx_node, idx_pred] = max(new_label, labels[idx_node, idx_pred])
#
#         if idx_node is None:
#             print(f"[WARN] Unmatched node for ({h}, {t})")
#
#     print(f"[generate_labels_and_mask] Total matched labels: {int(labels.sum())} / {len(dataset)}\n")
#     return torch.from_numpy(labels), torch.from_numpy(mask)
def to_bool_label(lab):
    """Robustly convert label to {0.0, 1.0} from str/bool/float/int."""
    if isinstance(lab, str):
        lab = lab.strip().lower()
        return 1.0 if lab in {"1", "true", "t", "yes"} else 0.0
    try:
        return 1.0 if float(lab) > 0.5 else 0.0
    except Exception:
        return 0.0

def generate_labels_and_mask(dataset, node_to_const_dict, const_to_node_dict, pred_dict):
    # --- optional debug ---
    if len(dataset) > 0:
        print("\n=== DEBUG: Checking label mapping ===")
        print(f"Sample from dataset: {dataset[0]}")
        print(f"Pred_dict keys (first 10): {list(pred_dict.keys())[:10]}")

    num_nodes, num_preds = len(node_to_const_dict), len(pred_dict)
    labels = np.zeros((num_nodes, num_preds), dtype=np.float32)
    mask   = np.zeros((num_nodes, num_preds), dtype=np.float32)

    for (h, r, t, lab) in dataset:
        if r == RDF_type_string:
            idx_node = const_to_node_dict.get(h, None)
            idx_pred = pred_dict.get(t, None)
        else:
            # binary fact -> pair node
            idx_node = const_to_node_dict.get((h, t), None)
            idx_pred = pred_dict.get(r, None)

        if idx_node is not None and idx_pred is not None:
            mask[idx_node, idx_pred] = 1.0
            labels[idx_node, idx_pred] = to_bool_label(lab)

    print(f"[generate_labels_and_mask] Total matched labels: {int(labels.sum())} / {len(dataset)}\n")
    return torch.from_numpy(labels), torch.from_numpy(mask)


# -------------------------------------------------------------------------
# Decode helper functions
# -------------------------------------------------------------------------
def decode(node_dict, num_binary, num_unary, binaryPredicates, unaryPredicates, feature_vectors, threshold):
    threshold_indices = torch.nonzero(feature_vectors >= threshold)
    decoded = set()
    for idx in threshold_indices:
        i, j = idx.tolist()
        const, pred = node_dict[i], j
        if isinstance(const, tuple) and pred < num_binary:
            decoded.add(f"{const[0]}\t{binaryPredicates[pred]}\t{const[1]}")
        elif not isinstance(const, tuple) and pred >= num_binary:
            decoded.add(f"{const}\trdf:type\t{unaryPredicates[pred - num_binary]}")
    return decoded


def decode_with_scores(examples, output, const_to_node_dict, pred_dict):
    scores = {}
    flat = normalize_to_binary(examples, is_test=True)
    for (h, r, t, _) in flat:
        if r == RDF_type_string:
            idx_node = const_to_node_dict.get(h, None)
            idx_pred = pred_dict.get(t, None)
        else:
            idx_node = const_to_node_dict.get((h, t), None)
            idx_pred = pred_dict.get(r, None)
        if idx_node is not None and idx_pred is not None:
            scores[(h, r, t)] = float(output[idx_node][idx_pred].item())
    return scores


# -------------------------------------------------------------------------
# Inference utilities
# -------------------------------------------------------------------------
def predict_entailed_fast(model, binaryPredicates, unaryPredicates, dataset, query_dataset,
                          max_iterations=1, threshold=0.5, device='cpu'):
    """Iteratively predict entailed triples."""
    num_binary, num_unary = len(binaryPredicates), len(unaryPredicates)
    all_entailed = set()
    unchanged, iteration = False, 1

    while not unchanged:
        print(f"GNN iteration {iteration}", end='\r')
        dataset_x, edge_index, edge_type, node2const, const2node, pred_dict, _ = encode_input_dataset(
            dataset, query_dataset, binaryPredicates, unaryPredicates)
        data = Data(x=dataset_x, edge_index=edge_index, edge_type=edge_type).to(device)
        entailed = decode(node2const, num_binary, num_unary, binaryPredicates,
                          unaryPredicates, model(data), threshold)
        if entailed.issubset(all_entailed):
            unchanged = True
            print("\nNo new entailed facts.")
        else:
            all_entailed |= entailed
            dataset |= entailed
            if max_iterations and iteration >= max_iterations:
                unchanged = True
        iteration += 1
    return all_entailed


def output_scores(model, binaryPredicates, unaryPredicates, incomplete_graph, examples,
                  device='cpu', add_2hop=True):
    """Compute prediction scores for given triples."""
    x, edge_index, edge_type, node2const, const2node, pred_dict, _ = encode_input_dataset(
        incomplete_graph, examples, binaryPredicates, unaryPredicates, add_2hop, is_test=True)
    model.eval()
    pred = model(Data(x=x, edge_index=edge_index, edge_type=edge_type).to(device))
    return decode_with_scores(examples, pred, const2node, pred_dict)


# -------------------------------------------------------------------------
# Misc utilities
# -------------------------------------------------------------------------
def split_known(triples, ratio=0.9):
    DATA_LENGTH = len(triples)
    candidate = np.arange(DATA_LENGTH)
    np.random.shuffle(candidate)
    idx_known = candidate[:int(DATA_LENGTH * ratio)]
    idx_unknown = candidate[int(DATA_LENGTH * ratio):]
    known = [triples[i] for i in idx_known]
    unknown = [triples[i] for i in idx_unknown]
    return known, unknown


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        return
    for filename in os.listdir(directory_path):
        path = os.path.join(directory_path, filename)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def save_important_files(config_file, exp_name):
    os.makedirs(f'experiments/{exp_name}/scripts', exist_ok=True)
    copyfile(config_file, f'experiments/{exp_name}/configs.yaml')
    for file in ['utils.py', 'train.py', 'gnn_architectures.py', 'evaluate.py', 'evaluate_rules.py']:
        if os.path.exists(file):
            copyfile(file, f'experiments/{exp_name}/scripts/{file}')


