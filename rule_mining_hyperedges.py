"""
rule_mining_hyperedges.py
----------------------------------------
Automatic generation of Source–Target (S→T) hyperedge rules
for n-ary KGs such as FB-AUTO, JF17K, WikiPeople.

Usage:
    python rule_mining_hyperedges.py
----------------------------------------
Author: Ziheng Yang (UCAS)
Date: 2025-７
"""

# import os
# import random
# from collections import defaultdict
# from itertools import combinations
#
# def mine_hyperedge_rules(dataset_path, output_path, min_conf=0.6, max_rules=200):
#     """
#     Automatically generate hyperedge-style rules (S→T)
#     from dataset train.txt file.
#
#     Each rule has format:
#         r1(A,B) ∧ r2(B,C) → r3(A,C)   confidence
#     """
#     triples = []
#     train_file = os.path.join(dataset_path, "train.txt")
#     if not os.path.exists(train_file):
#         print(f"⚠️ No train.txt found in {dataset_path}, skip.")
#         return
#
#     # Load triples
#     with open(train_file, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split("\t")
#             if len(parts) < 3:
#                 continue
#             h, r, t = parts[:3]
#             triples.append((h, r, t))
#
#     if len(triples) == 0:
#         print(f"⚠️ No valid triples found in {dataset_path}.")
#         return
#
#     # Co-occurrence mining
#     head_rules = defaultdict(lambda: defaultdict(int))
#     for (h1, r1, t1), (h2, r2, t2) in combinations(triples[:5000], 2):  # limit sample for speed
#         if t1 == h2:  # path composition
#             head_rules[(r1, r2, "path")]["count"] += 1
#         elif h1 == h2:  # shared head
#             head_rules[(r1, r2, "shared_head")]["count"] += 1
#         elif t1 == t2:  # shared tail
#             head_rules[(r1, r2, "shared_tail")]["count"] += 1
#
#     rules = []
#     for (r1, r2, typ), stat in head_rules.items():
#         conf = round(random.uniform(min_conf, 0.95), 2)
#         if typ == "path":
#             head = random.choice([r1, r2])
#             rule = f"{r1}(A,B) ∧ {r2}(B,C) → {head}(A,C)\t{conf}"
#         elif typ == "shared_head":
#             head = random.choice([r1, r2])
#             rule = f"{r1}(A,B) ∧ {r2}(A,C) → {head}(B,C)\t{conf}"
#         elif typ == "shared_tail":
#             head = random.choice([r1, r2])
#             rule = f"{r1}(A,B) ∧ {r2}(C,B) → {head}(A,C)\t{conf}"
#         else:
#             continue
#         rules.append(rule)
#
#     random.shuffle(rules)
#     rules = rules[:max_rules]
#
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         for r in rules:
#             f.write(r + "\n")
#
#     print(f"✅ Mined {len(rules)} hyperedge rules for {dataset_path} → {output_path}")
#
#
# def main():
#     data_root = "data"
#     if not os.path.exists(data_root):
#         print("❌ data/ directory not found. Please place this script in the project root.")
#         return
#
#     datasets = [
#         "FB-AUTO",
#         "JF17K",
#         "JF17K-3",
#         "JF17K-4",
#         "WikiPeople",
#         "WikiPeople-3",
#         "WikiPeople-4",
#     ]
#
#     for d in datasets:
#         path = os.path.join(data_root, d)
#         output = os.path.join(path, "rules_hyperedges.txt")
#         mine_hyperedge_rules(path, output)
#
#     print("\n🎯 All done. Rules generated under each dataset folder!")
#
#
# if __name__ == "__main__":
#     main()
import argparse
from collections import defaultdict, Counter
import itertools
import math
import os

def parse_args():
    ap = argparse.ArgumentParser("Role-aware path rule miner (v6): r1∘r2=>r3 with arg joins + PCA conf")
    ap.add_argument("--train_path", required=True, help="tab file: rel \\t arg1 \\t arg2 ...")
    ap.add_argument("--out_path", required=True, help="rules output txt")
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--min_conf", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--target_head", type=str, default=None, help="only keep rules ending with this head relation")
    ap.add_argument("--enable_3hop", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def load_facts(path):
    rel2facts = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) < 3:  # rel + >=2 args
                continue
            rel, args = parts[0], parts[1:]
            rel2facts[rel].append(tuple(args))
    return rel2facts

def build_indexes(rel2facts):
    """
    为每个关系建立：
      - role2ents[role_idx] -> set(entities)
      - pair2count[(role_i_entity, role_j_entity, i, j)] -> count（快速体量统计）
    以及全局 (ent -> positions) 反向索引，便于 join
    """
    rel2_role2ents = {}
    rel2_pairs_anypos = {}
    ent2pos = defaultdict(list)  # ent -> [(rel, role_idx)]
    for r, facts in rel2facts.items():
        max_len = max(len(f) for f in facts)
        role2ents = [set() for _ in range(max_len)]
        pairs_anypos = Counter()
        for f in facts:
            L = len(f)
            for i,e in enumerate(f):
                role2ents[i].add(e)
                ent2pos[e].append((r,i))
            # 所有成对角色组合（任意两位置）
            for i,j in itertools.combinations(range(L), 2):
                pairs_anypos[(f[i], f[j], i, j)] += 1
        rel2_role2ents[r] = role2ents
        rel2_pairs_anypos[r] = pairs_anypos
    return rel2_role2ents, rel2_pairs_anypos, ent2pos

def entity_pair_in_rel(rel2facts, rel, a, b):
    """检查在关系 rel 的任意两个角色位置是否存在同一事实使 (a,b) 同时出现（允许顺序任意）"""
    # 速度权衡：简单遍历；如需加速可为每个rel建 entity->fact ids map
    for f in rel2facts[rel]:
        if a in f and b in f:
            return True
    return False

def enumerate_2hop_rules(rel2facts, rel2_role2ents, ent2pos, min_support, min_conf, target_head, verbose):
    """
    模板：r1(s, …, j) ∧ r2(j', …, t) => r3(s, …, t)
    做法：
      1) 枚举 r1,r2；对 r1 的每个角色 j 与 r2 的每个角色 j' 做 join（同一实体出现）
      2) 取 r1 的源位 s ∈ roles(r1)，r2 的目标位 t ∈ roles(r2)，形成 candidate (a_s, b_t)
      3) 在所有 r3 中检查是否存在事实包含 (a_s, b_t) 这对实体（任意角色对）
      4) 统计 support/ conf / pca-conf
    """
    rules = []
    rels = list(rel2facts.keys())
    # 预先建立 各rel的role->entity集合，便于笛卡尔积过滤
    for r1, r2 in itertools.permutations(rels, 2):
        role2ents1 = rel2_role2ents[r1]
        role2ents2 = rel2_role2ents[r2]
        # 枚举 join 角色 (j from r1) 与 (j' from r2)
        for j in range(len(role2ents1)):
            for jp in range(len(role2ents2)):
                join_ents = role2ents1[j] & role2ents2[jp]
                if len(join_ents) < min_support:
                    continue
                # 对 r1 选一个源位 s, 对 r2 选一个目标位 t（可与 join位不同）
                for s in range(len(role2ents1)):
                    if s == j:  # 源位与join位可以不同，也可允许相同；此处允许相同会产生更多匹配，按需切换
                        pass
                    for t in range(len(role2ents2)):
                        # 构造 (a_s, b_t) 候选对：需要从事实级统计，避免仅用集合近似
                        # 遍历实体 join_ents，通过 ent2pos 找到具体事实组合
                        pair_counts = Counter()  # (a, b) -> count
                        body_support = 0
                        # 遍历出现于 r1@j 与 r2@jp 的同一实体 e
                        for e in join_ents:
                            # 找到 e 在 r1 的所有事实位置
                            r1_pos = [(rr, idx) for (rr, idx) in ent2pos[e] if rr == r1 and idx == j]
                            r2_pos = [(rr, idx) for (rr, idx) in ent2pos[e] if rr == r2 and idx == jp]
                            if not r1_pos or not r2_pos:
                                continue
                            # 遍历 r1 的所有事实，抽取源位实体 a_s
                            for f1 in rel2facts[r1]:
                                if len(f1) <= max(j, s):
                                    continue
                                if f1[j] != e:
                                    continue
                                a = f1[s]
                                # 遍历 r2 的所有事实，抽取目标位实体 b_t
                                for f2 in rel2facts[r2]:
                                    if len(f2) <= max(jp, t):
                                        continue
                                    if f2[jp] != e:
                                        continue
                                    b = f2[t]
                                    pair_counts[(a,b)] += 1
                                    body_support += 1
                        if body_support < min_support:
                            continue
                        # 寻找 head r3 使 (a,b) 共现
                        for r3 in rels:
                            if r3 == r1 or r3 == r2:
                                continue  # 去自循环
                            head_hits = 0
                            # 在 r3 的事实中验证 (a,b) 共同出现（任意角色对）
                            for (a,b), c in pair_counts.items():
                                if entity_pair_in_rel(rel2facts, r3, a, b):
                                    head_hits += c
                            if head_hits < min_support:
                                continue
                            # 计算 conf / pca-conf / lift
                            conf = head_hits / (body_support + 1e-9)
                            if conf < min_conf:
                                continue
                            # PCA-confidence 近似：用 r1在s位的实体可与任意r2的t位实体组合作为分母
                            denom_pca = 0
                            for e in role2ents1[s]:
                                # 该 e 作为源位，能与多少 r2@t 的实体配对？
                                denom_pca += len(role2ents2[t])
                            pca_conf = head_hits / (denom_pca + 1e-9)

                            score = conf * (1 + math.log(1 + pca_conf*10.0))
                            rules.append({
                                "r1": r1, "r2": r2, "r3": r3,
                                "s": s, "j": j, "jp": jp, "t": t,
                                "support": head_hits, "body": body_support,
                                "conf": round(conf, 4),
                                "pca": round(pca_conf, 6),
                                "score": round(score, 4)
                            })
                            if verbose:
                                print(f"[2hop] {r1}[{s}] & {r2}[{t}] via join {j}-{jp} -> {r3} | supp={head_hits} conf={conf:.3f} pca={pca_conf:.3e} score={score:.3f}")
    # 去重：同一 (r1,r2,r3,s,j,jp,t) 保留最高分
    uniq = {}
    for R in rules:
        key = (R["r1"], R["r2"], R["r3"], R["s"], R["j"], R["jp"], R["t"])
        if key not in uniq or R["score"] > uniq[key]["score"]:
            uniq[key] = R
    rules = sorted(uniq.values(), key=lambda x: -x["score"])
    # 目标关系过滤
    if target_head:
        rules = [R for R in rules if R["r3"] == target_head]
    return rules

def save_rules(rules, out_path, topk=200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# === Role-aware 2-hop rules: r1(arg_s, …, arg_j) ∧ r2(arg_j', …, arg_t) → r3(arg_s, …, arg_t)\n")
        for R in rules[:topk]:
            f.write(
                f"{R['r1']}[arg{R['s']}] & {R['r2']}[arg{R['t']}] "
                f"via join {R['r1']}[arg{R['j']}]={R['r2']}[arg{R['jp']}] "
                f"=> {R['r3']}(arg{R['s']}, arg{R['t']}) "
                f"support={R['support']} body={R['body']} conf={R['conf']} pca={R['pca']} score={R['score']}\n"
            )
    print(f"✅ Saved {min(len(rules), topk)} rules to {out_path}")

def main():
    args = parse_args()
    rel2facts = load_facts(args.train_path)
    print(f"Loaded {len(rel2facts)} relations from {args.train_path}")
    rel2_role2ents, rel2_pairs_anypos, ent2pos = build_indexes(rel2facts)
    rules = enumerate_2hop_rules(
        rel2facts, rel2_role2ents, ent2pos,
        min_support=args.min_support,
        min_conf=args.min_conf,
        target_head=args.target_head,
        verbose=args.verbose,
    )
    save_rules(rules, args.out_path, args.topk)

if __name__ == "__main__":
    main()
