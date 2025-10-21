# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Optional, Iterable, Any


# ------------------------------------------------------------------------------
# Core data structures
# ------------------------------------------------------------------------------

class Rule:
    """
    A generic rule with a head predicate and one/two body predicates.
    This is primarily used by legacy two-body pipelines (e.g., NCRL/DRUM).
    """
    def __init__(self, rule_id: str, head: str, body: List[str], conf: float, pred_k: Optional[Any] = None):
        self.id = rule_id
        self.head = head
        self.body = body
        self.conf = conf
        if pred_k is not None:
            self.pred_k = pred_k

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rule):
            return self.id == other.id and self.head == other.head and self.body == other.body
        return False

    def __hash__(self) -> int:
        return hash((self.id, self.head, tuple(self.body)))

    def __len__(self) -> int:
        return len(self.body)

    def __str__(self) -> str:
        return f"{self.id} {self.head} :- {self.body} conf: {self.conf}"


class MGNNRule:
    """
    A richer rule object that stores:
      - head: predicate name of the head
      - body: list of body predicate names
      - conf: confidence/weight
      - is_inverse: list[bool] flags (legacy, kept for compatibility). Length equals len(body).
      - arity: list[int] arities for each body predicate (and implicitly head if needed in parsing)
      - text: original textual form (optional)
    """
    def __init__(
        self,
        rule_id: str,
        head: str,
        body: List[str],
        conf: float,
        is_inverse: Iterable[bool],
        arity: Iterable[int],
        text: Optional[str] = None
    ):
        self.id = rule_id
        self.head = head
        self.body = list(body)
        self.conf = conf
        self.text = text

        # Support lists for variable-arity rules
        self.is_inverse = list(is_inverse) if is_inverse is not None else [False] * len(self.body)
        self.arity = list(arity) if arity is not None else [2] * len(self.body)

    def __len__(self) -> int:
        return len(self.body)

    def __repr__(self) -> str:
        return f"MGNNRule(id={self.id}, head={self.head}, body={self.body}, conf={self.conf}, arity={self.arity})"


class RuleSet:
    """
    RuleSet groups rules by head predicate and supports materialization
    (currently binary-focused) and a stub for n-ary reasoning.
    """
    def __init__(self, rules: List[Rule] | List[MGNNRule]):
        self.rules = rules
        self.rules_grouped_by_head: Dict[str, List[int]] = {}
        for i, rule in enumerate(rules):
            head = rule.head
            if head not in self.rules_grouped_by_head:
                self.rules_grouped_by_head[head] = []
            self.rules_grouped_by_head[head].append(i)

    # -------------------- Binary-focused materialization (legacy) --------------------
    def materialize(self, facts_with_conf: Dict[Tuple[str, str, str], float],
                    sub2obj: Dict[str, Iterable[str]],
                    obj2sub: Dict[str, Iterable[str]]) -> Dict[Tuple[str, str, str], float]:
        """
        Materialize new confidences for (h, r, t) triples using a small set of
        hard-coded rule templates (R1-R8) from the original codebase.

        facts_with_conf: mapping (h, r, t) -> confidence
        sub2obj / obj2sub: bipartite adjacency dicts for subjects/objects (binary graph)
        """
        con_pairs: Dict[str, set] = {}
        for x in sub2obj:
            objs = set(sub2obj[x])
            subs = set(obj2sub[x]) if x in obj2sub else set()
            con_pairs[x] = subs | objs
        for x in obj2sub:
            if x not in con_pairs:
                con_pairs[x] = set(obj2sub[x])

        output_facts_with_conf: Dict[Tuple[str, str, str], float] = {}

        for fact, conf in facts_with_conf.items():
            h, r, t = fact
            if conf == 1:
                output_facts_with_conf[fact] = conf
                continue

            # initialize with 0 and aggregate contributions
            new_conf = 0.0
            if r == 'rdf:type':
                # Unary head rules (R5-R8 in original code)
                if r' rdf:type ' in " ".join(fact):  # keep structure parity; safe noop
                    pass
                # For type heads we need to fetch by 't' as type
                head_type = t
                if head_type in self.rules_grouped_by_head:
                    for rule_index in self.rules_grouped_by_head[head_type]:
                        rule = self.rules[rule_index]
                        if getattr(rule, 'id', None) == 'R5':
                            #  <x rdf:type T> :- <x rdf:type B>
                            if (h, 'rdf:type', rule.body) in facts_with_conf:
                                body_conf = facts_with_conf[(h, 'rdf:type', rule.body)]
                                new_conf += body_conf * rule.conf
                        elif getattr(rule, 'id', None) == 'R6':
                            #  <x rdf:type T> :- exists y: <y rdf:type B> and y connected to x
                            for y in con_pairs.get(h, []):
                                if (y, 'rdf:type', rule.body) in facts_with_conf:
                                    body_conf = facts_with_conf[(y, 'rdf:type', rule.body)]
                                    new_conf += body_conf * rule.conf
                        elif getattr(rule, 'id', None) == 'R7':
                            #  <x rdf:type T> :- exists o: <x B o>
                            for obj in sub2obj.get(h, []):
                                if (h, rule.body, obj) in facts_with_conf:
                                    body_conf = facts_with_conf[(h, rule.body, obj)]
                                    new_conf += body_conf * rule.conf
                        elif getattr(rule, 'id', None) == 'R8':
                            #  <x rdf:type T> :- exists s: <s B x>
                            for sub in obj2sub.get(h, []):
                                if (sub, rule.body, h) in facts_with_conf:
                                    body_conf = facts_with_conf[(sub, rule.body, h)]
                                    new_conf += body_conf * rule.conf
            else:
                # Binary head rules (R1-R4 in original code)
                if r in self.rules_grouped_by_head:
                    for rule_index in self.rules_grouped_by_head[r]:
                        rule = self.rules[rule_index]
                        body_conf = 0.0
                        if getattr(rule, 'id', None) == 'R1':
                            # <h r t> :- <h b t>
                            if (h, rule.body, t) in facts_with_conf:
                                body_conf = facts_with_conf[(h, rule.body, t)]
                        elif getattr(rule, 'id', None) == 'R2':
                            # <h r t> :- <t b h> (inverse)
                            if (t, rule.body, h) in facts_with_conf:
                                body_conf = facts_with_conf[(t, rule.body, h)]
                        elif getattr(rule, 'id', None) == 'R3':
                            # <h r t> :- <h rdf:type B>
                            if (h, 'rdf:type', rule.body) in facts_with_conf:
                                body_conf = facts_with_conf[(h, 'rdf:type', rule.body)]
                        elif getattr(rule, 'id', None) == 'R4':
                            # <h r t> :- <t rdf:type B>
                            if (t, 'rdf:type', rule.body) in facts_with_conf:
                                body_conf = facts_with_conf[(t, 'rdf:type', rule.body)]
                        new_conf += body_conf * rule.conf

            output_facts_with_conf[fact] = min(1.0, new_conf)

        return output_facts_with_conf

    def materialize_nary(self, facts_with_conf_nary: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, ...], float]:
        """
        Placeholder for n-ary materialization. You can implement hyperedge reasoning here.

        facts_with_conf_nary: mapping (pred, arg1, arg2, ..., argK) -> confidence
        Suggested approach:
          - Unify variables across body predicates using a join over shared symbols (?A, ?B, ...).
          - For each rule, match assignments that satisfy all bodies, then write/update the head tuple.
          - Aggregate confidence with min/product and clamp to [0,1].
        """
        # For now, return the input unchanged.
        return facts_with_conf_nary

    def reasoning(self, facts_with_conf: Dict[Tuple[str, str, str], float],
                  sub2obj: Dict[str, Iterable[str]],
                  obj2sub: Dict[str, Iterable[str]],
                  num_steps: int = 2) -> Dict[Tuple[str, str, str], float]:
        """
        Iterate materialization a few steps for binary facts.
        """
        for _ in range(num_steps):
            facts_with_conf = self.materialize(facts_with_conf, sub2obj, obj2sub)
        return facts_with_conf


# ------------------------------------------------------------------------------
# Patterns / helpers for rule parsing
# ------------------------------------------------------------------------------

# Matches: <predicate>[?A,?B,?C]
_PATTERN_MGNN = re.compile(r'<(\S*?)>\[(.*?)\]')

# Backward compatibility helpers
li0  = ['?A', '?B', '?A', '?B']
li1  = ['?A', '?B', '?B', '?A']
li2  = ['?B', '?A', '?B', '?A']
li3  = ['?A', '?B', '?A', '?C']
li4  = ['?A', '?B', '?C', '?A']
li5  = ['?A', '?B', '?B', '?C']
li6  = ['?A', '?B', '?C', '?B']
li7  = ['?B', '?A', '?A', '?C']
li8  = ['?B', '?A', '?C', '?A']
li9  = ['?B', '?A', '?B', '?C']
li10 = ['?B', '?A', '?C', '?B']
li_all = [li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10]


# ------------------------------------------------------------------------------
# MGNN-style parser (extended to n-ary)
# ------------------------------------------------------------------------------

def rule_parser_mgnn(line: str, conf: float) -> MGNNRule:
    """
    Parse MGNN-style rule text.
    Examples:
      <r_head>[?A,?B] <= <r_body>[?A,?B]
      <head>[?A] <= <body1>[?A] ∧ <body2>[?A,?B]
      <head4>[?A,?B,?C,?D] <= <b1>[?A,?C,?B,?D]  (n-ary supported)

    Returns:
      MGNNRule with id that reflects the pattern family (legacy types kept),
      and with `arity` covering every body predicate. `is_inverse` is kept for
      compatibility but not required by newer n-ary pipelines.
    """
    result = _PATTERN_MGNN.findall(line)
    assert len(result) >= 2, f"Cannot parse rule line: {line}"

    # Head
    head_pred, head_vars_raw = result[0]
    head_vars = [v.strip() for v in head_vars_raw.split(',')] if head_vars_raw else []
    head_arity = len(head_vars)

    if len(result) == 2:
        # One body predicate
        body_pred, body_vars_raw = result[1]
        body_vars = [v.strip() for v in body_vars_raw.split(',')] if body_vars_raw else []
        body_arity = len(body_vars)

        # Legacy identification (R1/R2 vs type-based)
        rule_id = None
        is_inverse: List[bool] = []
        arity: List[int] = []

        if head_arity == 1:
            # <x type H> :- <x type B>  OR  <x type H> :- <x R y>
            # Keep legacy tags: R4/R5/...
            if body_arity == 2:
                # type <- binary
                if body_vars[0] == '?A':
                    rule_id = 'R4'  # head type from subject role
                    is_inverse = [False]
                else:
                    rule_id = 'R5'  # head type from object role (inverse)
                    is_inverse = [True]
                arity = [2]
            else:
                # type <- type
                rule_id = 'R6'  # generic type propagation
                is_inverse = [False]
                arity = [1]
        elif head_arity == 2:
            # binary head
            if body_arity == 2:
                if [head_vars[0], head_vars[1]] == [body_vars[0], body_vars[1]]:
                    rule_id = 'R1'
                    is_inverse = [False]
                elif [head_vars[0], head_vars[1]] == [body_vars[1], body_vars[0]]:
                    rule_id = 'R2'
                    is_inverse = [True]
                else:
                    rule_id = 'pattern-mismatch'
                    is_inverse = [False]
                arity = [2]
            else:
                # head is binary, body is unary -> simple typing (R3/R4 equiv)
                rule_id = 'R3orR4'
                is_inverse = [False]
                arity = [1]
        else:
            # n-ary head with single body: keep a generic id
            rule_id = f"nary_{1}"
            is_inverse = [False]
            arity = [body_arity]

        return MGNNRule(rule_id, head_pred, [body_pred], conf, is_inverse, arity, text=line)

    # >= 2 body predicates (including full n-ary)
    bodies = result[1:]
    body_preds = [bp for bp, _ in bodies]
    body_vars_list = [[v.strip() for v in bv.split(',')] if bv else [] for _, bv in bodies]
    arities = [len(bv) for bv in body_vars_list]

    # Keep legacy conj-pattern ids if head is unary/binary and both bodies are binary.
    rule_id = None
    is_inverse: List[bool] = [False] * len(body_preds)

    if head_arity == 1 and len(body_preds) == 2 and arities == [2, 2]:
        vars_li = [body_vars_list[0][0], body_vars_list[0][1], body_vars_list[1][0], body_vars_list[1][1]]
        vars_li_inv = [body_vars_list[1][0], body_vars_list[1][1], body_vars_list[0][0], body_vars_list[0][1]]
        for i, li in enumerate(li_all):
            if vars_li == li or vars_li_inv == li:
                rule_id = f"type-conj{i}"
                break

    if rule_id is None and head_arity == 2 and len(body_preds) == 2 and arities == [2, 2]:
        vars_li = [body_vars_list[0][0], body_vars_list[0][1], body_vars_list[1][0], body_vars_list[1][1]]
        vars_li_inv = [body_vars_list[1][0], body_vars_list[1][1], body_vars_list[0][0], body_vars_list[0][1]]
        for i, li in enumerate(li_all):
            if vars_li == li or vars_li_inv == li:
                rule_id = f"conj{i}"
                break

    # If we cannot classify into legacy families, assign a generic n-ary id
    if rule_id is None:
        rule_id = f"nary_{len(body_preds)}"

    return MGNNRule(rule_id, head_pred, body_preds, conf, is_inverse, arities, text=line)


# ------------------------------------------------------------------------------
# Additional parsers kept for compatibility (INDIGO / NCRL / DRUM / TXT)
# ------------------------------------------------------------------------------

def rule_parser_indigo(line: str, rule_id: str) -> Rule:
    """
    Parse a simple INDIGO-style rule row (kept as-is from your original code).
    """
    conf = 0.0
    head = None
    body = None
    if rule_id in ('pattern1', 'pattern2'):
        conf, body, head = line.strip().split('\t')
        body = [body]
    elif rule_id == 'pattern3':
        conf, head = line.strip().split('\t')
        body = [head]
    elif rule_id in ('pattern4', 'pattern5', 'pattern6'):
        conf, body1, body2, head = line.strip().split('\t')
        body = [body1, body2]
    return Rule(rule_id, head, body, float(conf))


def normalize_2(head: str, body1: str) -> Tuple[str, str]:
    """
    Normalize rule for inverse markers used in NCRL.
    """
    if 'inv_' in head:
        head = head[4:]
        body1 = 'inv_' + body1 if 'inv_' not in body1 else body1[4:]
    return head, body1


def normalize_3(head: str, body1: str, body2: str) -> Tuple[str, str, str]:
    """
    Normalize 2-body rule for inverse markers used in NCRL.
    """
    if 'inv_' in head:
        head = head[4:]
        body1 = 'inv_' + body1 if 'inv_' not in body1 else body1[4:]
        body2 = 'inv_' + body2 if 'inv_' not in body2 else body2[4:]
    return head, body1, body2


def rule_parser_ncrl(line: str, threshold: float) -> Optional[Rule]:
    """
    Parse a NCRL-style rule row. Drops rules with conf < threshold.
    """
    items = line.strip().split()
    conf = float(items[0])
    if conf < threshold:
        return None
    head = items[2]
    if ',' not in line:
        body1 = items[4]
        head, body1 = normalize_2(head, body1)
        if 'inv_' not in body1:
            rule_id = 'R1'
        else:
            rule_id = 'R2'
            body1 = body1[4:]
        return Rule(rule_id, head, [body1], conf)
    else:
        body1 = items[4][:-1]
        body2 = items[5]
        head, body1, body2 = normalize_3(head, body1, body2)
        is_inv = ['inv_' in body1, 'inv_' in body2]
        if is_inv == [False, False]:
            rule_id = 'cp1'
        elif is_inv == [False, True]:
            rule_id = 'cp2'
            body2 = body2[4:]
        elif is_inv == [True, False]:
            rule_id = 'cp3'
            body1 = body1[4:]
        else:
            rule_id = 'cp4'
            body1 = body1[4:]
            body2 = body2[4:]
        return Rule(rule_id, head, [body1, body2], conf)


def rule_parser_drum_mm(line: str) -> Rule:
    """
    Parse a DRUM multi-modal style rule line.
    """
    items = line.strip().split(' :  ')
    conf = float(items[2])
    head = items[0]
    body = items[1][1:-1].split(', ')
    if len(body) == 1:
        body1 = body[0].replace("'", "")
        if 'inv_' not in body1:
            rule_id = 'R1'
        else:
            rule_id = 'R2'
            body1 = body1[4:]
        return Rule(rule_id, head, [body1], conf)
    else:
        assert len(body) == 2
        body1 = body[0].replace("'", "")
        body2 = body[1].replace("'", "")
        is_inv = ['inv_' in body1, 'inv_' in body2]
        if is_inv == [False, False]:
            rule_id = 'cp1'
        elif is_inv == [False, True]:
            rule_id = 'cp2'
            body2 = body2[4:]
        elif is_inv == [True, False]:
            rule_id = 'cp3'
            body1 = body1[4:]
        else:
            rule_id = 'cp4'
            body1 = body1[4:]
            body2 = body2[4:]
        return Rule(rule_id, head, [body1, body2], conf)


# A simple TXT rule parser for lines like:
#   r1(x,y) ∧ r2(y,z) => r3(x,z) [0.82]
#   r4(x,y) => r5(x,y) [0.60]
_TXT_RULE = re.compile(r'([^\s\[\]]+)\s*(?:∧\s*([^\s\[\]]+))?\s*=>\s*([^\s\[\]]+)\s*\[([0-9.]+)\]')

def rule_parser_txt(line: str) -> Optional[Rule]:
    """
    Parse a simple logic text rule with confidence.
    Returns a generic Rule; for n-ary variants you should prefer rule_parser_mgnn().
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    m = _TXT_RULE.match(line)
    if not m:
        return None
    body1, body2, head, conf = m.groups()
    conf = float(conf)
    if body2:
        body = [body1, body2]
        rid = '2-body'
    else:
        body = [body1]
        rid = '1-body'
    return Rule(rid, head, body, conf)
