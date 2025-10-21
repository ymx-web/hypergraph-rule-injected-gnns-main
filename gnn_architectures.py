import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing


# -------------------------
# 1. Basic EC-GCN Layer
# -------------------------
class EC_GCNConv(MessagePassing):
    """Entity-Context Graph Convolutional Layer (baseline)."""

    def __init__(self, num_unary, num_binary, num_edge_types, num_singleton_nodes, aggr='add'):
        super(EC_GCNConv, self).__init__(aggr=aggr)

        self.num_features = num_unary + num_binary
        self.A = Parameter(torch.Tensor(self.num_features, self.num_features))
        self.B = Parameter(torch.Tensor(num_edge_types, self.num_features, self.num_features))
        self.bias_single = Parameter(torch.Tensor(self.num_features))
        self.bias_pair = Parameter(torch.Tensor(self.num_features))

        # Initialization
        self.A.data.normal_(0, 0.01)
        self.B.data.normal_(0, 0.01)
        self.bias_single.data.normal_(0, 0.001)
        self.bias_pair.data.normal_(0, 0.001)

        self.num_unary = num_unary
        self.num_binary = num_binary
        self.num_edge_types = num_edge_types
        self.num_singleton_nodes = num_singleton_nodes
        self.aggr = aggr

    def forward(self, x, edge_index, edge_type):
        """Standard relational message passing."""
        if self.aggr == 'add':
            out = F.linear(x, self.A, bias=None)

        # Aggregate messages by edge type
        for i in range(self.num_edge_types):
            edge_mask = edge_type == i
            temp_edges = edge_index[:, edge_mask]
            msg = self.propagate(temp_edges, x=x, size=(x.size(0), x.size(0)))
            if self.aggr == 'add':
                msg = F.linear(msg, self.B[i], bias=None)
                out += msg

        # Add biases
        out[:self.num_singleton_nodes] += self.bias_single
        out[self.num_singleton_nodes:] += self.bias_pair

        out = clipped_relu(out)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


# -------------------------
# 2. HyperRule Layer
# -------------------------
class HyperRuleLayer(nn.Module):
    """
    Structural rule propagation layer.
    It injects hyperedge-level signals derived from mined logical rules.

    Each hyperedge e = (S_e → T_e, β_e)
    - S_e: list of source relations
    - T_e: single target relation
    - β_e: rule confidence weight
    """

    def __init__(self, hidden_dim, use_gate=True):
        super(HyperRuleLayer, self).__init__()
        self.linear_msg = nn.Linear(hidden_dim, hidden_dim)
        self.linear_upd = nn.Linear(hidden_dim, hidden_dim)
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, he_tensors):
        """
        x: [num_relations, hidden_dim]
        he_tensors: dict from utils.pack_hyperedges
        """
        he_ptr = he_tensors["he_ptr"]
        he_src = he_tensors["he_src"]
        he_tgt = he_tensors["he_tgt"]
        he_w = he_tensors["he_w"]

        num_hyperedges = len(he_tgt)
        if num_hyperedges == 0:
            return self.linear_upd(x)

        # Aggregate messages for each hyperedge
        out = x.clone()
        for e in range(num_hyperedges):
            start = he_ptr[e].item()
            end = he_ptr[e + 1].item()
            src_ids = he_src[start:end]

            # Mean aggregation of source relation embeddings
            src_emb = x[src_ids].mean(dim=0)
            msg = self.linear_msg(src_emb)
            tgt = he_tgt[e].item()
            weight = he_w[e].item()

            if self.use_gate:
                gate_val = torch.sigmoid(self.gate(torch.cat([x[tgt], msg], dim=-1)))
                out[tgt] = x[tgt] + weight * gate_val * msg
            else:
                out[tgt] = x[tgt] + weight * msg

        out = self.linear_upd(out)
        out = clipped_relu(out)
        return out


# -------------------------
# 3. Combined GNN (EC-GCN + HyperRule)
# -------------------------
class HyperRuleGNN(nn.Module):
    """
    Two-stage reasoning model:
    (1) Standard EC-GCN propagation.
    (2) Hyperedge rule-guided structural enhancement.
    """

    def __init__(self, num_unary, num_binary, num_edge_types, num_singleton_nodes=0,
                 hidden_dim=None, num_layers=2, dropout=0.0, aggr='add', use_gate=True):
        super(HyperRuleGNN, self).__init__()

        self.base_gnn = GNN(num_unary, num_binary, num_edge_types,
                            num_singleton_nodes, num_layers, dropout, aggr)
        hidden_dim = num_unary + num_binary if hidden_dim is None else hidden_dim
        self.rule_layer = HyperRuleLayer(hidden_dim, use_gate=use_gate)
        self.dropout = dropout

    def forward(self, data, he_tensors=None):
        """
        Forward pass with optional hyperedge propagation.
        he_tensors can be None (for baseline training).
        """
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.base_gnn.conv1(x, edge_index, edge_type)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.base_gnn.conv2(x, edge_index, edge_type)

        if he_tensors is not None:
            x = self.rule_layer(x, he_tensors)

        assert torch.max(x).item() <= 1.0
        return x

    # ➕ NEW: expose rule parameter stacks for rule_loss()
    def rule_params(self):
        """
        Return parameter stacks used by rule-based loss.
        Shape:
          - para1: [1 + num_edge_types, F, F] from conv1 (A and all B[i])
          - para2: [1 + num_edge_types, F, F] from conv2
        """
        # conv1
        A1 = self.base_gnn.conv1.A           # [F, F]
        B1 = self.base_gnn.conv1.B           # [E, F, F]
        para1 = torch.cat([A1.unsqueeze(0), B1], dim=0)  # [1+E, F, F]

        # conv2
        A2 = self.base_gnn.conv2.A
        B2 = self.base_gnn.conv2.B
        para2 = torch.cat([A2.unsqueeze(0), B2], dim=0)

        return para1, para2


# -------------------------
# 4. Baseline two-layer GNN
# -------------------------
class GNN(nn.Module):
    """Baseline two-layer EC-GCN network."""

    def __init__(self, num_unary, num_binary, num_edge_types,
                 num_singleton_nodes=0, num_layers=2, dropout=0.0, aggr='add'):
        super(GNN, self).__init__()
        self.conv1 = EC_GCNConv(num_unary, num_binary, num_edge_types, num_singleton_nodes, aggr=aggr)
        self.conv2 = EC_GCNConv(num_unary, num_binary, num_edge_types, num_singleton_nodes, aggr=aggr)
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.conv1(x, edge_index, edge_type)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        assert torch.max(x).item() <= 1.0
        return x

    # ➕ NEW: expose rule parameter stacks for rule_loss()
    def rule_params(self):
        """
        Return parameter stacks used by rule-based loss.
        Shape:
          - para1: [1 + num_edge_types, F, F] from conv1 (A and all B[i])
          - para2: [1 + num_edge_types, F, F] from conv2
        """
        # conv1
        A1 = self.conv1.A           # [F, F]
        B1 = self.conv1.B           # [E, F, F]
        para1 = torch.cat([A1.unsqueeze(0), B1], dim=0)  # [1+E, F, F]

        # conv2
        A2 = self.conv2.A
        B2 = self.conv2.B
        para2 = torch.cat([A2.unsqueeze(0), B2], dim=0)

        return para1, para2


# -------------------------
# 5. Activation functions
# -------------------------
def modified_sigmoid(x, k=1, c=0):
    """Scaled sigmoid activation."""
    return 1 / (1 + torch.exp(-k * (x - c)))


def clipped_relu(x, a=1):
    """ReLU clipped between 0 and a."""
    return torch.clamp(x, 0, a)
