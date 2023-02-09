from abc import ABC

from .UUIILayer import UUIILayer

import torch
import torch_geometric
import numpy as np
import random
from torch_sparse import SparseTensor, mul, sum


class UUIIModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_ii_layers,
                 learning_rate,
                 embed_k,
                 l_w,
                 sim_ii,
                 random_seed,
                 name="UUII",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.n_ii_layers = num_ii_layers

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # similarity embeddings
        self.Gis = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gis.weight)
        self.Gis.to(self.device)

        # item-item graph
        self.sim_ii = self.compute_normalized_laplacian(sim_ii)

        # graph convolutional network for item-item graph
        propagation_network_ii_list = []
        for layer in range(self.n_ii_layers):
            propagation_network_ii_list.append((UUIILayer(), 'x, edge_index -> x'))
        self.propagation_network_ii = torch_geometric.nn.Sequential('x, edge_index', propagation_network_ii_list)
        self.propagation_network_ii.to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def compute_normalized_laplacian(adj):
        deg = sum(adj, dim=-1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    def propagate_embeddings(self):
        item_embeddings = self.Gis.weight.to(self.device)
        for layer in range(self.n_ii_layers):
            item_embeddings = list(self.propagation_network_ii.children())[layer](
                item_embeddings.to(self.device),
                self.sim_ii.to(self.device))
        return item_embeddings

    def forward(self, inputs, **kwargs):
        gu, gi, gis = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)
        gamma_i_s = torch.squeeze(gis).to(self.device)

        gamma_i_final = gamma_i + torch.nn.functional.normalize(gamma_i_s, 2)

        xui = torch.sum(gamma_u * gamma_i_final, 1)

        return xui, gamma_u, gamma_i, gamma_i_s

    def predict(self, start, end, gis, **kwargs):
        gi = self.Gi.weight + torch.nn.functional.normalize(gis, 2)
        return torch.matmul(self.Gu.weight[start:end].to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        gis = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos, gamma_i_s_pos = self.forward(inputs=(self.Gu.weight[user[:, 0]],
                                                                           self.Gi.weight[pos[:, 0]], gis[pos[:, 0]]))
        xu_neg, _, gamma_i_neg, gamma_i_s_neg = self.forward(inputs=(self.Gu.weight[user[:, 0]],
                                                                     self.Gi.weight[neg[:, 0]], gis[neg[:, 0]]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.mean(torch.nn.functional.softplus(-difference))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2) +
                                         gamma_i_s_pos.norm(2).pow(2) +
                                         gamma_i_s_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
