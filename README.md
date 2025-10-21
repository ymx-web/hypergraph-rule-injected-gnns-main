# CONSTRAINT-AWARE HYPERGRAPH REASONING VIA GENERATIVE RULE INJECTION IN KNOWLEDGE GRAPHS

This repository contains the **official implementation** of our ICASSP 2026 paper:  
**“Constraint-aware Hypergraph Reasoning via Generative Rule Injection in Knowledge Graphs.”**

---

## Overview

This work addresses the challenge of **multi-arity reasoning** in Knowledge Graph Completion (KGC),  
where relations often involve more than two entities (e.g., events, temporal or compositional facts).  
Traditional binary GNNs struggle to capture such high-order dependencies and symbolic constraints.

We propose a **Constraint-aware Hypergraph Reasoning** framework that unifies  
**Generative Rule Grammar (GRG)**–based symbolic representation and **neural hypergraph propagation**.  
Our approach enables both *rule-guided learning* and *explainable inference* in heterogeneous, constraint-rich KGs.

---

## Key Contributions

Grounded in the ICASSP 2026 paper, our work makes the following contributions:

1. **Generative Rule Grammar (GRG) for Unified Constraint Representation**  
   We introduce a *Generative Rule Grammar* that translates heterogeneous symbolic constraints—  
   including logical, weighted, temporal, spatial, and hierarchical rules—into a unified symbolic form.  
   This grammar enables expressive, high-order rule encoding and generative construction of new rule candidates.

2. **Source–Target (S–T) Hyperedge Formulation for Multi-entity Reasoning**  
   We reformulate symbolic rules as **Source–Target hyperedges** that connect multiple entities,  
   allowing executable reasoning beyond binary triples.  
   This design bridges symbolic dependencies and hypergraph message passing,  
   enabling scalable inference across mixed-arity KGs.

3. **Two-stage Constraint-aware GNN**  
   We develop a **two-stage propagation network** that couples relational aggregation  
   with rule-guided hyperedge propagation.  
   Confidence-aware weighting and gated fusion mechanisms jointly model factual and rule-based constraints,  
   improving both predictive accuracy and interpretability.

4. **Empirical Validation on Mixed-arity Benchmarks**  
   Experiments on **JF17K**, **WikiPeople**, and **FB-AUTO** datasets show consistent improvements  
   (average +0.02 MRR) over recent hypergraph and neural–symbolic baselines.  
   Ablation studies confirm the role of selective rule injection,  
   and case studies on a **real-world emergency KG** demonstrate interpretability  
   through transparent rule activation paths.

---

## Environment Setup

**Requirements**
```bash
python >= 3.9
torch >= 2.0
torch_geometric >= 2.5
numpy
pandas
pyyaml
scikit-learn
````

**Install dependencies**

```bash
pip install -r requirements.txt
```

---

##  Repository Structure

```
.
├── configs/                    # YAML configs for training and evaluation
│    ├── fbauto_v1_rules-train.yaml
│    ├── jf17k_v1_rules-train.yaml
│    ├── jf17k_v3_rules-train.yaml
│    └── ...
├── data/                       # Benchmark datasets
│    ├── FB-AUTO/
│    ├── JF17K/
│    ├── JF17K-3/
│    ├── JF17K-4/
│    ├── WikiPeople/
│    ├── WikiPeople-3/
│    └── WikiPeople-4/
│         ├── train.txt
│         ├── valid.txt
│         └── test.txt
├── predicates/                 # Predicate definition files (.csv)
│    └── WikiPeople-4_predicates.csv
├── rules/                      # Rule sets (mined or generative)
│    ├── FB-AUTO_rules.txt
│    ├── JF17K_rules.txt
│    └── WikiPeople_rules.txt
├── experiments/                # Saved checkpoints and logs
├── gnn_architectures.py        # Core hypergraph GNN layers
├── rule_reasoning.py           # Rule parsing, n-ary hyperedge reasoning
├── evaluate_rules.py           # Rule-based evaluation and extraction
├── train.py                    # Model training script
├── evaluate.py                 # Model evaluation script
└── utils.py                    # Auxiliary utilities
```

---

## Training

Train the model with constraint-aware rule guidance:

```bash
python train.py --config configs/KGC_WikiPeople-4-train.yaml --gpu 0
```

Models are saved under:

```
experiments/{exp_name}/models/model.pt
```

Snapshot checkpoints (`e{epoch}.pt`) are stored every 100 epochs.

---

##  Evaluation

Apply a trained model for hypergraph reasoning:

```bash
python evaluate.py --config configs/KGC_WikiPeople-4-eval.yaml --gpu 0
```

Outputs include:

* Link-prediction metrics (MRR, Hits@K)
* Constraint consistency rate
* Rule-guided confidence distributions

---

## Rule Generation and Injection

Our model integrates **Generative Rule Injection (GRI)** for neural–symbolic alignment.

**Example usage:**

```bash
python evaluate_rules.py
```

**Rule format:**

```
r1(x,y) ∧ r2(y,z) => r3(x,z) [0.82]
r4(x,y,z) => r5(x,z,y) [0.74]
```

Rules are parsed and transformed into trainable hyperedge priors in
`rule_reasoning.py`, and injected dynamically during GNN propagation.

---

## Supported Datasets

| Dataset                                      | #Entities | #Relations | Arity | Description                                                |
| -------------------------------------------- | --------- | ---------- | ----- | ---------------------------------------------------------- |
| **FB-AUTO**                                  | ~3.5K      | 8        | 2,4,5   | Automotive knowledge (models, engines, parts)              |
| **JF17K / JF17K-3 / JF17K-4**                | ~28K      |322         | 2–6   | Event-centric person–organization–place graph              |
| **WikiPeople / WikiPeople-3 / WikiPeople-4** | ~47K      | 60         | 2–9   | People–event–organization knowledge derived from Wikipedia |

---

##  Model Highlights

| Component                           | Function                                                              |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Generative Rule Grammar (GRG)**   | Encodes and synthesizes constraints from mixed rule types             |
| **Source–Target Hyperedge Mapping** | Converts symbolic rules into executable hyperedges                    |
| **Constraint-aware GNN**            | Two-stage relational and hyperedge propagation with confidence fusion |
| **Rule Reasoning Engine**           | Applies symbolic constraints dynamically during inference             |
| **Rule Extraction Module**          | Recovers interpretable symbolic dependencies from learned GNN weights |

---

## Reproduction

You can reproduce all experimental results by running:

```bash
python train.py --config configs/{dataset}-train.yaml
python evaluate.py --config configs/{dataset}-eval.yaml
```

---

##  Citation

If you use this work, please cite:

```bibtex
@inproceedings{yang2026constraint,
  title     = {Constraint-aware Hypergraph Reasoning via Generative Rule Injection in Knowledge Graphs},
  author    = {Yang, Mengxue  and Xin, Jindou  and Zhang, Jingqi and Zhang, Xiaruo and Yang,Ziyi and Li, Ying},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year      = {2026},
  organization = {IEEE}
}
```

---

## Acknowledgement

This repository **extends our prior work**
[“Rule-guided GNNs for Explainable Knowledge Graph Reasoning” (AAAI 2025)](https://arxiv.org/abs/2501.12345)
by generalizing from **binary rule guidance** to **constraint-aware multi-arity hypergraph reasoning**.

We thank collaborators from **UCAS** and **Institute of Software, Chinese Academy of Sciences**
for their continued support and insightful discussions.

**Related prior work citation:**

```bibtex
@inproceedings{wang2025rule, title={Rule-guided GNNs for Explainable Knowledge Graph Reasoning}, 
author={Wang, Zhe and Ma, Suxue and Wang, Kewen and Zhuang Zhiqiang}, 
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, year={2025}
}

```
```
@article{HyCubE,
  title={HyCubE: Efficient Knowledge Hypergraph 3D Circular Convolutional Embedding}, 
  author={Li, Zhao and Wang, Xin and Zhao, Jun and Guo, Wenbin and Li, Jianxin},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  year={2025},
  volume={37},
  number={4},
  pages={1902--1914},
  publisher={IEEE}
}

Li Z, Wang X, Zhao J, et al. HyCubE: Efficient Knowledge Hypergraph 3D Circular Convolutional Embedding[J]. IEEE Transactions on Knowledge and Data Engineering, 2025, 37(4): 1902-1914.
```


```


