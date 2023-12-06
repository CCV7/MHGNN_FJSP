import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class MLP(nn.Module):
    def __init__(self, num_layers, op_input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(op_input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(op_input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLPActor(nn.Module):
    def __init__(self, num_layers, op_input_dim, hidden_dim, output_dim):

        super(MLPActor, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(op_input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()


            self.linears.append(nn.Linear(op_input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, op_input_dim, hidden_dim, output_dim):

        super(MLPCritic, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(op_input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(op_input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)

class GNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 op_input_dim,
                 mch_input_dim,
                 hidden_dim,
                 learn_eps,
                 nighbor_pooling_type,
                 device):

        super(GNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.nighbor_pooling_type = nighbor_pooling_type
        self.learn_eps = learn_eps
        self.mlps = torch.nn.ModuleList()
        self.bn = torch.nn.BatchNorm1d(op_input_dim)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, op_input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_nighbor_list=None, Adj_block=None):
        if self.nighbor_pooling_type == "max":

            pooled = self.maxpool(h, padded_nighbor_list)
        else:

            pooled = torch.mm(Adj_block, h)
            if self.nighbor_pooling_type == "average":
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_nighbor_list=None, Adj_block=None):

        if self.nighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_nighbor_list)
        else:
            pooled = torch.mm(Adj_block, h)
            if self.nighbor_pooling_type == "average":
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def forward(self,
                x,
                graph_pool,
                padded_ni,
                adj):

        x_concat = x
        graph_pool = graph_pool

        if self.nighbor_pooling_type == "max":
            padded_nighbor_list = padded_ni
        else:
            Adj_block = adj

        h = x_concat

        for layer in range(self.num_layers - 1):
            if self.nighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_nighbor_list=padded_nighbor_list)
            elif not self.nighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.nighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_nighbor_list=padded_nighbor_list)
            elif not self.nighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

        h_nodes = h.clone()
        pooled_h = torch.sparse.mm(graph_pool, h)
        return pooled_h, h_nodes

def random_a1(p, sequent):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    action = []
    log_a = dist.log_prob(s)

    for i in range(s.size(0)):
        a = sequent[i][s[i]]
        action.append(a)
    action = torch.stack(action,0)
    return action, s,log_a


def random_a2(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()

    log_a = dist.log_prob(s)

    return s,log_a

def greedy_a(p,sequent):

    _, index = p.squeeze(-1).max(1)
    action = []
    for i in range(index.size(0)):
        a = sequent[i][index[i]]
        action.append(a)
    action = torch.stack(action, 0)
    return action

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        encoder_transform = self.W1(encoder_outputs)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        return u_i


class MHGNNEncoder(nn.Module):
    def __init__(self,num_layers, num_mlp_layers, op_input_dim, hidden_dim, learn_eps, nighbor_pooling_type, device):
        super(MHGNNEncoder,self).__init__()
        self.feature_extract = GNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers,
                                        op_input_dim=op_input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        nighbor_pooling_type=nighbor_pooling_type,
                                        device=device).to(device)
    def forward(self,x,graph_pool, padded_ni, adj,):
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_ni=padded_ni,
                                                 adj=adj)

        return h_pooled,h_nodes


class Operation_Actor(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 nighbor_pooling_type,
                 op_input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(Operation_Actor, self).__init__()
        self.n_j = n_j
        self.device=device
        self.bn = torch.nn.BatchNorm1d(op_input_dim).to(device)
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        self.feature_extract = GNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        op_input_dim=op_input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        nighbor_pooling_type=nighbor_pooling_type,
                                        device=device).to(device)
        self.encoder = MHGNNEncoder(num_layers=num_layers,
                               num_mlp_layers=num_mlp_layers_feature_extract,
                               op_input_dim=op_input_dim,
                               hidden_dim=hidden_dim,
                               learn_eps=learn_eps,
                               nighbor_pooling_type=nighbor_pooling_type,
                               device=device).to(device)
        self._input = nn.Parameter(torch.Tensor(2 * hidden_dim))
        self._input.data.uniform_(-1, 1)
        self.actor = MLPActor(3, hidden_dim * 2, hidden_dim, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_ni,
                adj,
                sequent,
                mask,
                mask_m,
                dur,
                a_index,
                last_a,
                last=True,
                greedy=False
                ):

        h_pooled, h_nodes = self.encoder(x=x,
                                         graph_pool=graph_pool,
                                         padded_ni=padded_ni,
                                         adj=adj)

        if last:
            dummy = sequent.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            bne = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            sequent_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(sequent_feature)
            concateFea = torch.cat((sequent_feature, h_pooled_repeated), dim=-1)
            sequent_scos = self.actor(concateFea)
            sequent_scos = sequent_scos * 10
            mask_reshape = mask.reshape(sequent_scos.size())
            sequent_scos[mask_reshape] = float('-inf')
            pi = F.softmax(sequent_scos, dim=1)
            if greedy:
                action = greedy_a(pi,sequent)
                log_a = 0
                index = 0
            else:
                action, index, log_a = random_a1(pi, sequent)

            action1 = action.type(torch.long).to(self.device)

            b_x = dur.reshape(dummy.size(0), -1, self.n_m).to(self.device)

            mask_m = mask_m.reshape(dummy.size(0), -1, self.n_m)

            mma = torch.gather(mask_m, 1,
                                           action1.unsqueeze(-1).unsqueeze(-1).expand(mask_m.size(0), -1,
                                                                                      mask_m.size(2)))
            a_f = torch.gather(bne, 1,
                                          action1.unsqueeze(-1).unsqueeze(-1).expand(bne.size(0), -1,
                                                                                     bne.size(2))).squeeze(1)
            a_n = torch.gather(b_x, 1,
                                       action1.unsqueeze(-1).unsqueeze(-1).expand(b_x.size(0), -1,
                                                                                  b_x.size(2))).squeeze()  # [:,:-2]

            return action,index, log_a, a_n.detach(), a_f.detach(), mma.detach(), h_pooled.detach()

        else:
            dummy = sequent.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))

            bne = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)

            sequent_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(sequent_feature)
            concateFea = torch.cat((sequent_feature, h_pooled_repeated), dim=-1)
            sequent_scos = self.actor(concateFea)

            sequent_scos = sequent_scos.squeeze(-1) * 10
            mask_reshape = mask.reshape(sequent_scos.size())
            sequent_scos[mask_reshape] = float('-inf')

            pi = F.softmax(sequent_scos, dim=1)
            dist = Categorical(pi)

            log_a = dist.log_prob(a_index.to(self.device))
            entropy = dist.entropy()
            action1 = last_a.type(torch.long).cuda()

            b_x = dur.reshape(dummy.size(0), self.n_j*self.n_m, -1).to(self.device)

            mask_m = mask_m.reshape(dummy.size(0), -1, self.n_m)

            mma = torch.gather(mask_m, 1,
                                           action1.unsqueeze(-1).unsqueeze(-1).expand(mask_m.size(0), -1,
                                                                                      mask_m.size(2)))
            a_f = torch.gather(bne, 1,
                                          action1.unsqueeze(-1).unsqueeze(-1).expand(bne.size(0), -1,
                                                                                     bne.size(2))).squeeze(1)
            a_n = torch.gather(b_x, 1,
                                       action1.unsqueeze(-1).unsqueeze(-1).expand(b_x.size(0), -1,
                                                                                  b_x.size(2))).squeeze()  # [:,:-2]

            v = self.critic(h_pooled)

            return entropy, v, log_a, a_n.detach(), a_f.detach(), mma.detach(), h_pooled.detach()


class Machine_Actor(nn.Module):
    def __init__(self,n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 nighbor_pooling_type,
                 mch_input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 device):
        super(Machine_Actor,self).__init__()
        self.n_j = n_j
        self.bn = torch.nn.BatchNorm1d(hidden_dim).to(device)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim).to(device)
        self.n_m = n_m
        self.hidden_size=hidden_dim
        self.n_ops_perjob = n_m
        self.device = device
        self.fc2 = nn.Linear(2, hidden_dim, bias=False).to(device)
        self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)


        self.fc = nn.Linear(1,hidden_dim).to(device)
        self.fc1 = nn.Linear(1, hidden_dim).to(device)
        self.x0 = torch.zeros(hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim).to(device)
        self.fc4 = nn.Linear(hidden_dim * 2, hidden_dim).to(device)
        self.critic = MLPCritic(3, hidden_dim*2, hidden_dim, 1).to(device)
    def forward(self,a_n,hx,mma,m_time,m_a=None,last_hh=None,policy=False):
        feature = torch.cat([m_time.unsqueeze(-1), a_n.unsqueeze(-1)], -1)
        a_n = self.bn(self.fc2(feature).reshape(-1, self.hidden_size)).reshape(-1,self.n_m,self.hidden_size)
        pool = a_n.mean(dim=1)
        h_pooled_repeated = pool.unsqueeze(1).expand_as(a_n)
        pooled_repeated = hx.unsqueeze(1).expand_as(a_n)
        concateFea = torch.cat((a_n, h_pooled_repeated, pooled_repeated), dim=-1)
        m_scos = self.actor(concateFea)
        m_scos = m_scos.squeeze(-1) * 10
        m_scos = m_scos.masked_fill(mma.squeeze(1).bool(), float("-inf"))
        pi_m = F.softmax(m_scos, dim=1)

        if policy:
            pools = torch.cat([pool,hx],-1)
            v = self.critic(pools)
        else:
            v = 0
        return pi_m,v,last_hh
