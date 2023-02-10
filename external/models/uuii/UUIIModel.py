from abc import ABC

import torch
import torch_geometric
from torch_geometric.nn import LGConv
import numpy as np
import random
from torch_sparse import mul, sum


class UUIIModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_ii_layers,
                 num_ui_layers,
                 learning_rate,
                 embed_k,
                 l_w,
                 sim_ii,
                 adj,
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
        self.n_ui_layers = num_ui_layers

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

        # user-item graph
        self.adj = adj

        # graph convolutional network for item-item graph
        propagation_network_ii_list = []
        for layer in range(self.n_ii_layers):
            propagation_network_ii_list.append((LGConv(normalize=False), 'x, edge_index -> x'))
        self.propagation_network_ii = torch_geometric.nn.Sequential('x, edge_index', propagation_network_ii_list)
        self.propagation_network_ii.to(self.device)

        # graph convolutional network for user-item graph
        propagation_network_ui_list = []
        for layer in range(self.n_ui_layers):
            propagation_network_ui_list.append((LGConv(normalize=False), 'x, edge_index -> x'))
        self.propagation_network_ui = torch_geometric.nn.Sequential('x, edge_index', propagation_network_ui_list)
        self.propagation_network_ui.to(self.device)

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
        gis = self.Gis.weight.to(self.device)
        for layer in range(self.n_ii_layers):
            gis = list(self.propagation_network_ii.children())[layer](
                gis.to(self.device),
                self.sim_ii.to(self.device))

        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(self.n_ui_layers):
            all_embeddings += [torch.nn.functional.normalize(list(
                self.propagation_network_ui.children()
            )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device)), p=2, dim=1)]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu, gi + torch.nn.functional.normalize(gis.to(self.device), p=2, dim=1)

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.mean(torch.nn.functional.softplus(-difference))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
