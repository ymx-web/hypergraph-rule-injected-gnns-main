import os
import yaml
import argparse
import logging

import torch
import numpy as np

from utils import load_predicates, output_scores


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def read_triples(fn):
    """Read plain triples (h, r, t) from a text file where each line is 'h<TAB>r<TAB>t' or 'h r t'."""
    triples = []
    with open(fn, 'r') as f:
        for buf in f:
            parts = buf.strip().split()
            if len(parts) < 3:
                # also tolerate TSV
                parts = buf.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[:3]
                triples.append((h, r, t))
    return triples


def read_triples_with_scores(fn):
    """Read triples with numeric scores from file: 'h r t score' (space or tab)."""
    scores = {}
    with open(fn, 'r') as f:
        for buf in f:
            parts = buf.strip().split()
            if len(parts) < 4:
                parts = buf.strip().split('\t')
            if len(parts) >= 4:
                h, r, t, s = parts[:4]
                try:
                    scores[(h, r, t)] = float(s)
                except ValueError:
                    continue
    return scores


# ----------------------------------------------------------------------
# KGC ranking metrics (tail prediction): MRR / Hits@K
# ----------------------------------------------------------------------
def kg_ranking_metrics(triples, scores, truths, ks=(1, 3, 10)):
    """
    Compute MRR / Hits@K for tail prediction.
    Group by (h, r), rank candidate tails by predicted score, and measure ranks of tails with truth==1.

    Args:
        triples: List[(h,r,t)] in the evaluation file (one candidate per line).
        scores:  Dict[(h,r,t)] -> float
        truths:  Dict[(h,r,t)] -> float (0/1)

    Returns:
        mrr, hits@1, hits@3, hits@10
    """
    # group candidate tails by (h,r)
    groups = {}
    for (h, r, t) in triples:
        groups.setdefault((h, r), []).append(t)

    mrrs, hits = [], {k: [] for k in ks}

    for (h, r), tails in groups.items():
        # collect predictions & truth labels for this (h,r)
        sc = []
        gt = []
        idx_map = []

        for t in tails:
            if (h, r, t) not in scores or (h, r, t) not in truths:
                continue
            sc.append(scores[(h, r, t)])
            gt.append(1 if truths[(h, r, t)] > 0 else 0)
            idx_map.append(t)

        if len(sc) == 0:
            continue

        sc = np.array(sc, dtype=np.float32)
        gt = np.array(gt, dtype=np.int32)

        # if no positives in this query, skip
        pos_idx = np.where(gt == 1)[0]
        if pos_idx.size == 0:
            continue

        # rank in descending order of scores
        order = np.argsort(-sc)  # indices sorted by score desc

        # compute ranks for all positives
        for pi in pos_idx:
            # rank is 1-based position of pi in 'order'
            rank = int(np.where(order == pi)[0][0]) + 1
            mrrs.append(1.0 / rank)
            for k in ks:
                hits[k].append(1.0 if rank <= k else 0.0)

    if len(mrrs) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mrr = float(np.mean(mrrs))
    hits1 = float(np.mean(hits[1]))
    hits3 = float(np.mean(hits[3]))
    hits10 = float(np.mean(hits[10]))
    return mrr, hits1, hits3, hits10


# ----------------------------------------------------------------------
# Scoring with model
# ----------------------------------------------------------------------
def evaluate(eval_model, test_graph_path, test_examples_path, scores_path,
             binaryPredicates, unaryPredicates, device, add_2hop=True):
    """
    Use trained model to score all test examples against the incomplete test graph.
    Writes 'h\tr\t\tscore' to scores_path and also returns the dict for further metrics.
    """
    # Incomplete background graph
    incomplete_dataset = []
    with open(test_graph_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[:3]
                incomplete_dataset.append((h, r, t))

    # Candidate triples to score
    examples_dataset = read_triples(test_examples_path)

    # Call utils.output_scores (works with your current utils.py)
    scores = output_scores(
        eval_model,
        binaryPredicates,
        unaryPredicates,
        incomplete_dataset,
        examples_dataset,
        device=device,
        add_2hop=add_2hop
    )

    # Write out
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, 'w') as out:
        for (h, r, t), s in scores.items():
            out.write(f"{h}\t{r}\t{t}\t{s}\n")

    return scores


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="model")  # model file name without .pt prefix
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = configs["exp_name"]
    dataset = configs["dataset_name"]
    add_2hop = configs.get("add_2hop", True)

    # Paths
    test_graph_path = configs["test_graph"]  # incomplete background graph file (h r t)
    test_example_dir = f"data/{dataset}/test"  # contains test{i}.txt and test{i}_with_truth_values.txt
    model_path = f"experiments/{exp_name}/models/{args.model}.pt"

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(f"experiments/{exp_name}/runs", exist_ok=True)
    handler = logging.FileHandler(f"experiments/{exp_name}/runs/eval_result.txt")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    # Predicates
    binaryPredicates, unaryPredicates = load_predicates(dataset)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    # Load model
    model = torch.load(model_path, map_location=device).to(device)
    model.eval()

    # Aggregate metrics over 10 folds (test0..test9)
    all_mrr, all_h1, all_h3, all_h10 = [], [], [], []

    for i in range(10):
        test_examples_path = os.path.join(test_example_dir, f"test{i}.txt")
        truths_path = os.path.join(test_example_dir, f"test{i}_with_truth_values.txt")
        scores_path = f"experiments/{exp_name}/scores-{i}.txt"

        # 1) score with model
        scores = evaluate(model,
                          test_graph_path,
                          test_examples_path,
                          scores_path,
                          binaryPredicates,
                          unaryPredicates,
                          device,
                          add_2hop=add_2hop)

        # 2) load triples and truths, then compute ranking metrics
        triples = read_triples(test_examples_path)
        truths = read_triples_with_scores(truths_path)

        mrr, h1, h3, h10 = kg_ranking_metrics(triples, scores, truths)
        all_mrr.append(mrr)
        all_h1.append(h1)
        all_h3.append(h3)
        all_h10.append(h10)

        logger.info(f"Fold {i}: MRR={mrr:.4f}, Hits@1={h1:.4f}, Hits@3={h3:.4f}, Hits@10={h10:.4f}")
        print(f"Fold {i}: MRR={mrr:.4f}, Hits@1={h1:.4f}, Hits@3={h3:.4f}, Hits@10={h10:.4f}")

    # Report mean metrics across folds
    mean_mrr = float(np.mean(all_mrr)) if all_mrr else 0.0
    mean_h1 = float(np.mean(all_h1)) if all_h1 else 0.0
    mean_h3 = float(np.mean(all_h3)) if all_h3 else 0.0
    mean_h10 = float(np.mean(all_h10)) if all_h10 else 0.0

    print(f"MRR={mean_mrr:.4f}, Hits@1={mean_h1:.4f}, Hits@3={mean_h3:.4f}, Hits@10={mean_h10:.4f}")
    logger.info(f"MRR={mean_mrr:.4f}, Hits@1={mean_h1:.4f}, Hits@3={mean_h3:.4f}, Hits@10={mean_h10:.4f}")
