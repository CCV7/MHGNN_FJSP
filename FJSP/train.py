from models.sdac import Operation_Actor, Machine_Actor
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
import os
from Env import FJSP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import csv
import copy
from Gantt_graph import Gantt
from Gen_instance import Dataset

device = torch.device(configs.device)


def Validate(vali_set, batch_size, policy_jo, policy_mc):
    op_select_policy = copy.deepcopy(policy_jo)
    mch_alloc_policy = copy.deepcopy(policy_mc)
    op_select_policy.eval()
    mch_alloc_policy.eval()

    def eval_model_bat(bat,i):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()

            env = FJSP(n_j=configs.n_j, n_m=configs.n_m)
            gantt_chart = Gantt( configs.n_j, configs.n_m)
            device = torch.device(configs.device)
            graph_pool_step = graph_pool(Type=configs.Type,
                                     batch_size=torch.Size(
                                         [batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=device)

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)

            j = 0

            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool=None
            while True:
                env_adj = Agr_ob(deepcopy(adj).to(device).to_sparse(), configs.n_j * configs.n_m)
                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
                env_mask = torch.from_numpy(np.copy(mask)).to(device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
                action, a_idx, log_a, action_node, _, mask_mch_action, hx = op_select_policy(x=env_fea,
                                                                                               graph_pool=graph_pool_step,
                                                                                               padded_ni=None,
                                                                                               adj=env_adj,
                                                                                               candidate=env_candidate
                                                                                               , mask=env_mask
                                                                                               , mask_mch=env_mask_mch
                                                                                               , dur=env_dur
                                                                                               , a_index=0
                                                                                               , old_action=0
                                                                                                ,mch_pool=pool
                                                                                               ,old_policy=True,
                                                                                                T=1
                                                                                               ,greedy=True
                                                                                               )

                pi_mch,pool = mch_alloc_policy(action_node, hx, mask_mch_action, env_mch_time)
                _, mch_a = pi_mch.squeeze(-1).max(1)
                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a,gantt_chart)
                j += 1
                if env.done():
                    break
            cost = env.mchsETs.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)
    all_time = torch.cat([eval_model_bat(bat,i) for i,bat in enumerate(vali_set)], 0)

    return all_time


def graph_pool(Type, batch_size, n_nodes, device):

    if Type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()
    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse.FloatTensor(idx, elem,
                                          torch.Size([batch_size[0],
                                                      n_nodes*batch_size[0]])
                                          ).to(device)
    return graph_pool


def Init_weight(net, scheme='orthogonal'):
    for e in net.parameters():
        if scheme == 'orthogonal':
            if len(e.size()) >= 2:
                nn.init.orthogonal_(e)

        elif scheme == 'normal':
            nn.init.normal(e, std=1e-2)
        elif scheme == 'xavier':
            nn.init.xavier_normal(e)


def Adv_nor(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs


def Agr_ob(obs_mb, n_node):
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch


def random_a2(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    log_a = dist.log_prob(s)
    return s,log_a


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.sequent_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.job_logprobs = []
        self.mch_logprobs = []
        self.mm = []
        self.first_task = []
        self.pre_task = []
        self.action = []
        self.mch = []
        self.dur = []
        self.mch_time = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.sequent_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.job_logprobs[:]
        del self.mch_logprobs[:]
        del self.mm[:]
        del self.first_task[:]
        del self.pre_task[:]
        del self.action[:]
        del self.mch[:]
        del self.dur[:]
        del self.mch_time[:]


class SDAC:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 softdate,
                 n_j,
                 n_m,
                 num_layers,
                 nighbor_pooling_type,
                 op_input_dim,
                 mch_input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.softdate = softdate
        self.k_epochs = k_epochs

        self.op_select_policy = Operation_Actor(n_j=configs.n_j,
                                    n_m=configs.n_m,
                                    num_layers=configs.num_layers,
                                    learn_eps=False,
                                    nighbor_pooling_type=configs.nighbor_pooling_type,
                                    op_input_dim=configs.op_input_dim,
                                    hidden_dim=configs.hidden_dim,
                                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                                    num_mlp_layers_critic=num_mlp_layers_critic,
                                    hidden_dim_critic=hidden_dim_critic,
                                    device=device)
        self.mch_alloc_policy = Machine_Actor(n_j=configs.n_j,
                                    n_m=configs.n_m,
                                    num_layers=configs.num_layers,
                                    learn_eps=False,
                                    nighbor_pooling_type=configs.nighbor_pooling_type,
                                    mch_input_dim=configs.mch_input_dim,
                                    hidden_dim=configs.hidden_dim,
                                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                                    device=device)

        self.last_op_sel_policy = deepcopy(self.op_select_policy)
        self.last_mch_alloc_policy = deepcopy(self.mch_alloc_policy)

        self.last_op_sel_policy.load_state_dict(self.op_select_policy.state_dict())
        self.last_mch_alloc_policy.load_state_dict(self.mch_alloc_policy.state_dict())

        self.job_optimizer = torch.optim.Adam(self.op_select_policy.parameters(), lr=lr)
        self.mch_optimizer = torch.optim.Adam(self.mch_alloc_policy.parameters(), lr=lr)

        self.job_scheduler = torch.optim.lr_scheduler.StepLR(self.job_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        self.mch_scheduler = torch.optim.lr_scheduler.StepLR(self.mch_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)


        self.MSE = nn.MSELoss()

    def update(self,  memories, epoch):
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        rewards_all_env = []

        for i in range(configs.batch_size):
            rewards = []

            discounted_reward = 0
            for reward, is_terminal in zip(reversed((memories.r_mb[0][i]).tolist()),
                                           reversed(memories.done_mb[0][i].tolist())):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0)
        for _ in range(configs.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            graph_pool_step = graph_pool(Type=configs.Type,
                                     batch_size=torch.Size(
                                         [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=device)

            job_log_prob = []
            mch_log_prob = []
            val = []
            m_a =None
            last_hh = None
            entropies = []
            job_entropy = []
            mch_entropies = []
            job_scheduler = LambdaLR(self.job_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            mch_scheduler = LambdaLR(self.mch_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            job_log_old_prob = memories.job_logprobs[0]
            mch_log_old_prob = memories.mch_logprobs[0]
            env_mm = memories.mm[0]
            env_dur = memories.dur[0]
            first_task = memories.first_task[0]
            pool=None
            for i in range(len(memories.fea_mb)):
                env_fea = memories.fea_mb[i]
                env_adj = memories.adj_mb[i]
                env_sequent = memories.sequent_mb[i]
                env_mask = memories.mask_mb[i]


                a_index = memories.a_mb[i]
                env_mch_time = memories.mch_time[i]

                old_action = memories.action[i]
                last_mch = memories.mch[i]

                a_entropy, v, log_a, a_n, _, mma, hx = self.op_select_policy(x=env_fea,
                                                                                           graph_pool=graph_pool_step,
                                                                                           padded_ni=None,
                                                                                           adj=env_adj,
                                                                                           sequent=env_sequent
                                                                                           , mask=env_mask
                                                                                           , mm=env_mm
                                                                                           , dur=env_dur
                                                                                           , a_index=a_index
                                                                                           , old_action=old_action
                                                                                           ,mch_pool=pool
                                                                                           , last=False
                                                                                           )
                pi_mch,pool = self.mch_alloc_policy(a_n, hx, mma, env_mch_time,m_a,last_hh,policy=True)
                val.append(v)
                dist = Categorical(pi_mch)
                log_mch = dist.log_prob(last_mch)
                mch_entropy = dist.entropy()

                job_entropy.append(a_entropy)
                mch_entropies.append(mch_entropy)

                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)

            job_log_prob, job_log_old_prob = torch.stack(job_log_prob, 0).permute(1, 0), torch.stack(job_log_old_prob,
                                                                                                     0).permute(1, 0)
            mch_log_prob, mch_log_old_prob = torch.stack(mch_log_prob, 0).permute(1, 0), torch.stack(mch_log_old_prob,
                                                                                                     0).permute(1, 0)
            val = torch.stack(val, 0).squeeze(-1).permute(1, 0)
            job_entropy = torch.stack(job_entropy, 0).permute(1, 0)
            mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)

            job_loss_sum = 0
            mch_loss_sum = 0
            for j in range(configs.batch_size):
                job_ratios = torch.exp(job_log_prob[j] - job_log_old_prob[j].detach())
                mch_ratios = torch.exp(mch_log_prob[j] - mch_log_old_prob[j].detach())
                advantages = rewards_all_env[j] - val[j].detach()
                advantages = Adv_nor(advantages)
                job_surr1 = job_ratios * advantages
                job_surr2 = torch.clamp(job_ratios, 1 - self.softdate, 1 + self.softdate) * advantages
                job_v_loss = self.MSE(val[j], rewards_all_env[j])
                job_loss = -1*torch.min(job_surr1, job_surr2) + 0.5*job_v_loss - 0.01 * job_entropy[j]
                job_loss_sum += job_loss


                mch_surr1 = mch_ratios * advantages
                mch_surr2 = torch.clamp(mch_ratios, 1 - self.softdate, 1 + self.softdate) * advantages
                mch_loss = -1*torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j]
                mch_loss_sum += mch_loss

            self.job_optimizer.zero_grad()
            job_loss_sum.mean().backward(retain_graph=True)


            self.last_op_sel_policy.load_state_dict(self.op_select_policy.state_dict())
            self.mch_optimizer.zero_grad()
            mch_loss_sum.mean().backward(retain_graph=True)
            self.job_optimizer.step()
            self.mch_optimizer.step()

            self.last_mch_alloc_policy.load_state_dict(self.mch_alloc_policy.state_dict())
            
            if configs.decayflag:
                self.job_scheduler.step()
            if configs.decayflag:
                self.mch_scheduler.step()

            return job_loss_sum.mean().item(), mch_loss_sum.mean().item()




def main(epochs):

    log = []
    graph_pool_step = graph_pool(Type=configs.Type,
                             batch_size=torch.Size(
                                 [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device=device)

    sdac = SDAC(configs.lr, configs.gamma, configs.k_epochs, configs.softdate,
              n_j=configs.n_j,
              n_m=configs.n_m,
              num_layers=configs.num_layers,
              nighbor_pooling_type=configs.nighbor_pooling_type,
              op_input_dim=configs.op_input_dim,
              mch_input_dim=configs.op_input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    train_dataset = Dataset(configs.n_j, configs.n_m, configs.low, configs.upper, configs.num_ins, 200)
    validat_dataset = Dataset(configs.n_j, configs.n_m, configs.low, configs.upper, 128, 200)

    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    record = 1000
    for epoch in range(epochs):
        memory = Memory()
        sdac.last_op_sel_policy.train()
        sdac.last_mch_alloc_policy.train()
        times, losses, rewards2, critic_rewards = [], [], [], []
        start = time.time()
        costs = []
        losses, rewards, critic_loss = [], [], []
        for batch_idx, batch in enumerate(data_loader):
            env = FJSP(configs.n_j, configs.n_m)
            data = batch.numpy()
            adj, fea, sequent, mask, mm, dur, mch_time, job_time = env.reset(data)

            job_log_prob = []
            mch_log_prob = []
            r_mb = []
            done_mb = []
            first_task = []
            pretask = []
            j = 0
            m_a = None
            last_hh = None
            pool = None
            ep_rewards = - env.initQuality
            env_mm = torch.from_numpy(np.copy(mm)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            while True:

                env_adj = Agr_ob(deepcopy(adj).to(device).to_sparse(), configs.n_j * configs.n_m)
                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_sequent = torch.from_numpy(np.copy(sequent)).long().to(device)

                env_mask = torch.from_numpy(np.copy(mask)).to(device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)

                action, a_idx, log_a, a_n, _, mma, hx = sdac.last_op_sel_policy(x=env_fea,
                                                                                               graph_pool=graph_pool_step,
                                                                                               padded_ni=None,
                                                                                               adj=env_adj,
                                                                                               sequent=env_sequent
                                                                                               , mask=env_mask

                                                                                               , mm=env_mm
                                                                                               , dur=env_dur
                                                                                               , a_index=0
                                                                                               , old_action=0
                                                                                               , mch_pool=pool
                                                                                               )

                pi_mch,pool = sdac.last_mch_alloc_policy(a_n, hx, mma, env_mch_time,m_a,last_hh)
                print(action,m_a)
                m_a, log_mch = random_a2(pi_mch)
                job_log_prob.append(log_a)

                mch_log_prob.append(log_mch)

                memory.mch.append(m_a)
                memory.pre_task.append(pretask)
                memory.adj_mb.append(env_adj)
                memory.fea_mb.append(env_fea)
                memory.sequent_mb.append(env_sequent)
                memory.action.append(deepcopy(action))
                memory.mask_mb.append(env_mask)
                memory.mch_time.append(env_mch_time)
                memory.a_mb.append(a_idx)

                adj, fea, reward, done, sequent, mask, job, _, mch_time, job_time = env.step(action.cpu().numpy(),
                                                                                               m_a)
                ep_rewards += reward

                r_mb.append(deepcopy(reward))
                done_mb.append(deepcopy(done))

                j += 1
                if env.done():
                    break
            memory.dur.append(env_dur)
            memory.mm.append(env_mm)
            memory.first_task.append(first_task)
            memory.job_logprobs.append(job_log_prob)
            memory.mch_logprobs.append(mch_log_prob)
            memory.r_mb.append(torch.tensor(r_mb).float().permute(1, 0))
            memory.done_mb.append(torch.tensor(done_mb).float().permute(1, 0))
           
            ep_rewards -= env.posRewards
           
            loss, v_loss = sdac.update(memory,batch_idx)
            memory.clear_memory()
            mean_reward = np.mean(ep_rewards)
            log.append([batch_idx, round(-mean_reward,2)])

            if batch_idx % 100 == 0:
                file_writing_obj = open(
                    f'./train_log/lr' + str(configs.n_j) + '_' + str(configs.n_m)+'_'+ str(configs.softdate)+'.txt', 'w')
                file_writing_obj.write(str(log))
            rewards.append(np.mean(ep_rewards).item())
            losses.append(loss)
            critic_loss.append(v_loss)

            cost = env.METS.max(-1).max(-1)
            costs.append(cost.mean())
            step = 20
            with open('batch_idx_loss2_1280_6_6.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['batch_idx', 'loss'])

                for i, loss in enumerate(losses):
                    writer.writerow([i + 1, loss])
            filepath = 'network'
            if (batch_idx + 1) % step  == 0 :
                end = time.time()
                times.append(end - start)
                start = end
                mean_loss = np.mean(losses[-step:])
                mean_reward = np.mean(costs[-step:])
                critic_losss = np.mean(critic_loss[-step:])
                filename = '{}'.format(str(configs.n_j)+ 'x' + str(configs.n_m))
                filepath = os.path.join(filepath, filename)
                epoch_dir = os.path.join(filepath, '%s_%s' % (10, batch_idx))
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                op_path = os.path.join(epoch_dir, '{}.pth'.format('op_select_policy'))
                mch_path = os.path.join(epoch_dir, '{}.pth'.format('mch_alloc_policy'))
                torch.save(sdac.op_select_policy.state_dict(), op_path)
                torch.save(sdac.mch_alloc_policy.state_dict(), mch_path)
                print('  batch %d/%d, reward: %2.3f, loss: %2.4f,time: %2.4fs' % (batch_idx, len(data_loader), mean_reward, mean_loss, times[-1]))
                t4 = time.time()
                vali_log = Validate(valid_loader, configs.batch_size, sdac.op_select_policy, sdac.mch_alloc_policy).mean()
                if vali_log<record:
                    epoch_dir = os.path.join(filepath, 'best_value')
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir)
                    op_path = os.path.join(epoch_dir, '{}.pth'.format('op_select_policy'))
                    mch_path = os.path.join(epoch_dir, '{}.pth'.format('mch_alloc_policy'))
                    torch.save(sdac.op_select_policy.state_dict(), op_path)
                    torch.save(sdac.mch_alloc_policy.state_dict(), mch_path)
                    record = vali_log
                print('vali:', vali_log)




if __name__ == '__main__':
    total1 = time.time()
    main(3)
    total2 = time.time()
    #print(total2 - total1)
