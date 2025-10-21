import os
import yaml
import argparse
import logging
import random
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np

from utils import (
    load_predicates,
    encode_input_dataset,
    generate_labels_and_mask,
    setup_seed,
    clear_directory,
    save_important_files,
    split_known,
)
from gnn_architectures import GNN
from evaluate_rules import cal_sc_for_guiding


# ==========================================================
# 约束 & 结构化掩码
# ==========================================================
def add_constraints(model):
    """Clamp parameters within [0,1] and enforce self-loops on A."""
    for _, p in model.named_parameters():
        p.data.clamp_(0, 1)
    with torch.no_grad():
        for conv_name in ["conv1", "conv2"]:
            conv = getattr(model, conv_name, None)
            if conv is not None and hasattr(conv, "A"):
                conv.A.fill_diagonal_(1.0)


def add_mask(model, num_binary):
    """结构化掩码：清零不允许的块，避免跨类型干扰。"""
    with torch.no_grad():
        for conv_name in ["conv1", "conv2"]:
            conv = getattr(model, conv_name, None)
            if conv is None:
                continue
            if hasattr(conv, "A"):
                conv.A[:num_binary, num_binary:] = 0
                conv.A[num_binary:, :num_binary] = 0
            if hasattr(conv, "B"):
                for i in range(conv.num_edge_types):
                    conv.B[i, :num_binary, :num_binary] = 0
                    conv.B[i, num_binary:, num_binary:] = 0
            if hasattr(conv, "bias_single"):
                conv.bias_single[:num_binary] = 0
            if hasattr(conv, "bias_pair"):
                conv.bias_pair[num_binary:] = 0


# ==========================================================
# 规则正则（安全版）
# ==========================================================
def rule_loss(model,
              sc_rl1, sc_rl2, mask_rl1,
              index_body1, index_body2,
              threshold_b, device):
    """
    - 未提供规则或没有模型暴露 rule_params() -> 返回 0
    - 否则将参数矩阵强度（mean(abs)）与规则置信度 L2 对齐
    """
    def as_tensor(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)

    sc_rl1 = as_tensor(sc_rl1)
    sc_rl2 = as_tensor(sc_rl2)

    if (sc_rl1 is None or sc_rl1.numel() == 0) and (sc_rl2 is None or sc_rl2.numel() == 0):
        return torch.tensor(0.0, device=device)

    if not hasattr(model, "rule_params"):
        return torch.tensor(0.0, device=device)

    para1, para2 = model.rule_params()
    loss = torch.tensor(0.0, device=device)

    if para1 is not None and sc_rl1 is not None and sc_rl1.numel() > 0:
        strength1 = para1.abs().mean(dim=tuple(range(1, para1.dim())))  # [T]
        n = min(strength1.numel(), sc_rl1.numel())
        if n > 0:
            loss = loss + ((strength1[:n] - sc_rl1[:n]) ** 2).mean()

    if para2 is not None and sc_rl2 is not None and sc_rl2.numel() > 0:
        strength2 = para2.abs().mean(dim=tuple(range(1, para2.dim())))  # [T]
        n = min(strength2.numel(), sc_rl2.numel())
        if n > 0:
            loss = loss + float(threshold_b) * ((strength2[:n] - sc_rl2[:n]) ** 2).mean()

    return loss


# ==========================================================
# 读取 FB-AUTO 样式数据
# ==========================================================
# def read_dataset(path):
#     """
#     FB-AUTO：每行 tab 分隔：
#       - 3列: r \t h \t t  （默认 label=1）
#       - 4列: r \t h \t t \t label
#     统一返回 (h, r, t, label)
#     """
#     dataset = []
#     with open(path, 'r') as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) == 3:
#                 r, h, t = parts
#                 label = 1.0
#             elif len(parts) == 4:
#                 r, h, t, label = parts
#                 label = float(label)
#             else:
#                 continue
#             dataset.append((h, r, t, label))
#     return dataset
# def read_dataset(path):
#     """
#     通用读取函数，兼容：
#     - JF17K / FB15K 风格：每行关系后跟若干实体（2~6个）
#     - FB-AUTO 风格：高元关系（>=4列）
#     - 若最后一列为数值，则视为 label，否则 label=1.0
#     返回统一格式：list[(head, relation, tail, label)]
#     """
#     dataset = []
#     with open(path, 'r') as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) < 3:
#                 continue  # 无效行
#
#             # 检查最后一列是否为浮点型标签
#             try:
#                 label_val = float(parts[-1])
#                 has_label = True
#             except ValueError:
#                 has_label = False
#
#             if has_label:
#                 # 显式带标签的行（例如: r h t 0.9）
#                 r = parts[0]
#                 ents = parts[1:-1]
#                 label = label_val
#             else:
#                 # 普通多元关系行
#                 r = parts[0]
#                 ents = parts[1:]
#                 label = 1.0
#
#             # 如果只有一个实体，跳过
#             if len(ents) < 2:
#                 continue
#
#             # 生成所有 pair 形式 (hi, hj)
#             # 例如 relation + (h, t) 组合
#             head = ents[0]
#             for tail in ents[1:]:
#                 dataset.append((head, r, tail, label))
#
#     print(f"[read_dataset] Loaded {len(dataset)} tuples from {path}")
#     return dataset
def read_dataset(path):
    """
    通用读取函数，自动适配：
    - FB-AUTO / JF17K / WikiPeople 等多元格式
    - 若最后一列为浮点数 -> 视为 label，否则 label=1.0
    - 对每行高元关系生成若干 (h, r, t, label) 三元组
    """
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue  # 无效行
            rel = parts[0]
            ents = parts[1:]

            # 判断是否有数值标签
            try:
                label_val = float(ents[-1])
                label = label_val
                ents = ents[:-1]  # 移除label列
            except ValueError:
                label = 1.0

            # 过滤掉空值或非法实体
            ents = [e for e in ents if e and not e.startswith('+') and not e.startswith('http')]

            if len(ents) < 2:
                continue  # 无有效 pair

            # 构造 (head, relation, tail) 三元组
            head = ents[0]
            for tail in ents[1:]:
                dataset.append((head, rel, tail, label))

    print(f"[read_dataset] Loaded {len(dataset)} tuples from {path}")
    return dataset




# ==========================================================
# 评估：负采样（filtered setting）
# ==========================================================
def _all_entities_from_const2node(const_to_node_dict):
    """从 const_to_node_dict 的键中收集所有实体（非 pair）。"""
    return [c for c in const_to_node_dict.keys() if not isinstance(c, tuple)]


def _score_triplet(pred_np, h, r, t, const_to_node_dict, pred_dict):
    """从预测矩阵中提取得分；若该 pair 不在图中，返回 None（用于过滤）。"""
    node_id = const_to_node_dict.get((h, t), None)
    pred_id = pred_dict.get(r, None)
    if node_id is None or pred_id is None:
        return None
    return float(pred_np[node_id, pred_id])


def eval_link_prediction(pred_matrix,
                         triples_valid,
                         const_to_node_dict,
                         pred_dict,
                         known_triples=None,
                         num_neg=500,
                         filtered=True,
                         sample_max=300):
    """
    双向负采样 + 过滤评估
    - pred_matrix: torch.Tensor [N, P]（模型输出，已在[0,1]）
    - triples_valid: list[(h,r,t,label=1.0)] 验证正例
    - known_triples: set[(h,r,t)] 需要过滤的真实事实（train+valid）
    """
    pred_np = pred_matrix.detach().cpu().numpy()
    entities = _all_entities_from_const2node(const_to_node_dict)
    known_set = set(known_triples) if known_triples else set()

    rr, h1, h3, h10 = [], [], [], []

    pool = triples_valid if len(triples_valid) <= sample_max else random.sample(triples_valid, sample_max)
    for (h, r, t, _) in pool:
        pos = _score_triplet(pred_np, h, r, t, const_to_node_dict, pred_dict)
        if pos is None:
            # 如果该正例对应的 pair 不在图中，跳过
            continue

        scores = [pos]

        # tail corrupt
        got = 0
        tries = 0
        while got < num_neg and tries < num_neg * 10:
            corrupt_t = random.choice(entities)
            tries += 1
            if filtered and (h, r, corrupt_t) in known_set:
                continue
            sc = _score_triplet(pred_np, h, r, corrupt_t, const_to_node_dict, pred_dict)
            if sc is None:
                continue  # 图中无该 pair，换一个
            scores.append(sc)
            got += 1

        # head corrupt
        got = 0
        tries = 0
        while got < num_neg and tries < num_neg * 10:
            corrupt_h = random.choice(entities)
            tries += 1
            if filtered and (corrupt_h, r, t) in known_set:
                continue
            sc = _score_triplet(pred_np, corrupt_h, r, t, const_to_node_dict, pred_dict)
            if sc is None:
                continue
            scores.append(sc)
            got += 1

        # 若负样本太少（图太稀），跳过该正例，避免不稳定
        if len(scores) < 3:
            continue

        # 排名（加入极小噪声防并列）
        arr = np.array(scores) + np.random.normal(0, 1e-6, size=len(scores))
        rank = int(np.where(np.argsort(-arr) == 0)[0][0]) + 1

        rr.append(1.0 / rank)
        h1.append(rank <= 1)
        h3.append(rank <= 3)
        h10.append(rank <= 10)

    if not rr:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(rr)), float(np.mean(h10)), float(np.mean(h3)), float(np.mean(h1))


# ==========================================================
# 单轮训练
# ==========================================================
def train_epoch(model, optimizer, scheduler, bce_loss,
                batch_data, train_mask,
                rule_pack, configs, device):
    """
    只计算训练损失；验证在主循环里做（带负采样）。
    """
    model.train()
    optimizer.zero_grad()

    out = model(batch_data)               # 模型输出已在 [0,1]（clipped_relu）
    out_sig = torch.sigmoid(out)          # 再过一次 sigmoid 更稳当（冗余但安全）
    train_pred = out_sig * train_mask.to(device)

    loss1 = bce_loss(train_pred, batch_data.y)

    # 规则正则（可选）
    sc_rl1, sc_rl2, mask_rl1, index_body1, index_body2 = rule_pack
    if configs.get("add_rules", False):
        loss2 = rule_loss(
            model, sc_rl1, sc_rl2, mask_rl1, index_body1, index_body2,
            configs.get("threshold_b", 0.0), device
        )
        loss = loss1 + float(configs.get("rule_weight", 0.0)) * loss2
    else:
        loss = loss1

    loss.backward()
    optimizer.step()
    scheduler.step()

    if configs.get("add_constraints", True):
        add_constraints(model)

    return float(loss.item())


# ==========================================================
# main
# ==========================================================
if __name__ == "__main__":
    setup_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = configs["exp_name"]
    os.makedirs(f"experiments/{exp_name}/models", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/runs", exist_ok=True)
    os.makedirs(f"experiments/{exp_name}/scripts", exist_ok=True)
    clear_directory(f"experiments/{exp_name}/runs")
    save_important_files(args.config, exp_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"experiments/{exp_name}/runs/log.txt")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    # Predicates
    binaryPredicates, unaryPredicates = load_predicates(configs["dataset_name"])
    num_binary, num_unary = len(binaryPredicates), len(unaryPredicates)
    print(f"Loaded {num_binary+num_unary} predicates "
          f"({num_binary} binary, {num_unary} unary) for {configs['dataset_name']}")

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset_full = read_dataset(configs["train_path"])
    valid_dataset = read_dataset(configs["valid_path"])

    # 将 train 拆为已知图 vs 预测查询，避免泄露
    known_ratio = float(configs.get("known_ratio", 0.9))
    train_known, train_query = split_known(train_dataset_full, ratio=known_ratio)

    # 构图：只用已知图；查询=valid + train_query
    graph_known = train_known
    graph_query = valid_dataset + train_query

    (x, edge_index, edge_types,
     node2const, const2node,
     pred_dict, num_singleton_nodes) = encode_input_dataset(
        graph_known,
        graph_query,
        binaryPredicates,
        unaryPredicates,
        add_2hop=bool(configs.get("add_2hop", True))
    )

    # 监督只在各自集合
    train_y, train_mask = generate_labels_and_mask(
        train_dataset_full, node2const, const2node, pred_dict
    )
    valid_labels, valid_mask = generate_labels_and_mask(
        valid_dataset, node2const, const2node, pred_dict
    )

    print("\n=== DEBUG (after encoding) ===")
    print("Train triples:", len(train_dataset_full), "Valid triples:", len(valid_dataset))
    print("train_y sum:", float(train_y.sum().item()), "train_mask sum:", float(train_mask.sum().item()))
    print("valid_labels sum:", float(valid_labels.sum().item()), "valid_mask sum:", float(valid_mask.sum().item()))
    print("X shape:", tuple(x.shape), "Preds:", len(pred_dict))

    # Graph Data
    data = Data(x=x, y=train_y, edge_index=edge_index, edge_type=edge_types).to(device)
    loader = DataLoader(dataset=[data], batch_size=1)

    # Model
    model = GNN(num_unary, num_binary, num_edge_types=4,
                num_singleton_nodes=num_singleton_nodes,
                num_layers=int(configs.get("num_layers", 2)),
                dropout=float(configs.get("dropout", 0.1))).to(device)

    # Optimizer / Scheduler
    lr = float(configs.get("learning_rate", 1e-3))
    wd = float(configs.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(configs.get("lr_gamma", 0.99)))

    # Constraints / Mask
    if configs.get("add_constraints", True):
        add_constraints(model)
    if configs.get("add_mask", True):
        add_mask(model, num_binary)

    # BCE weight（正例加权）
    bce_weight = float(configs.get("bce_weight", 5.0))
    weight = torch.where(train_y.to(device) == 1,
                         torch.tensor(bce_weight, device=device),
                         torch.tensor(1.0, device=device))
    bce_loss = torch.nn.BCELoss(reduction="mean", weight=weight)

    # 规则分数（可选）
    sc_rl1, sc_rl2, mask_rl1, index_body1, index_body2 = cal_sc_for_guiding(
        configs["train_path"],
        configs["dataset_name"],
        configs.get("threshold_a", 0.0),
        rule_file_path=configs.get("rule_file_path", None),
    )
    to_dev = lambda x: x if x is None else (x.to(device) if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32, device=device))
    rule_pack = (to_dev(sc_rl1), to_dev(sc_rl2), to_dev(mask_rl1), to_dev(index_body1), to_dev(index_body2))

    # 准备 filtered 评估所需：已知事实全集（train_full + valid）
    known_true = set((h, r, t) for (h, r, t, lab) in train_dataset_full if lab == 1.0)
    known_true.update((h, r, t) for (h, r, t, lab) in valid_dataset if lab == 1.0)

    # 训练循环
    logger.info("Training started...")
    best_mrr, best_epoch = 0.0, 0
    min_loss, bad_iter = None, 0
    max_bad = int(configs.get("early_stop_patience", 200))
    num_epochs = int(configs.get("num_epochs", 1200))
    log_every = int(configs.get("log_every", 50))

    for epoch in range(num_epochs):
        batch = next(iter(loader))
        loss = train_epoch(
            model, optimizer, scheduler, bce_loss,
            batch, train_mask.to(device),
            rule_pack, configs, device
        )

        # 评估（负采样 + 过滤）
        model.eval()
        with torch.no_grad():
            pred_full = torch.sigmoid(model(batch))
            mrr, h10, h3, h1 = eval_link_prediction(
                pred_matrix=pred_full,
                triples_valid=valid_dataset,
                const_to_node_dict=const2node,
                pred_dict=pred_dict,
                known_triples=known_true,
                num_neg=int(configs.get("num_neg_eval", 50)),
                filtered=False,
                sample_max=int(configs.get("valid_sample_max", 400)),
            )

        # 早停/保存
        if min_loss is None or loss < min_loss:
            min_loss, bad_iter = loss, 0
        else:
            bad_iter += 1

        if mrr > best_mrr:
            best_mrr, best_epoch = mrr, epoch
            torch.save(model, f"experiments/{exp_name}/models/best_model.pt")

        if epoch % log_every == 0:
            msg = (f"Epoch {epoch:04d} | Loss {loss:.5f} | "
                   f"MRR {mrr:.4f} | Hits@10 {h10:.4f} | Hits@3 {h3:.4f} | Hits@1 {h1:.4f}")
            print(msg)
            logger.info(msg)

        if bad_iter > max_bad:
            print(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Best epoch: {best_epoch}, Best MRR: {best_mrr:.4f}")
    torch.save(model, f"experiments/{exp_name}/models/model_final.pt")
