from copy import deepcopy
from Env import FJSP
import copy
import torch
import os
from train import Agr_ob
from Gen_instance import Dataset
import numpy as np
import time
from Params import configs
from torch.utils.data import DataLoader
from train import SDAC


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


def getdata(filename='./pubilc_dataset'):

    f=open(filename,'r')
    line=f.readline()
    line_data=line.split()
    num_float=list(map(float,line_data))
    n=int(num_float[0])
    m=int(num_float[1])
    ops_mchs={}
    ops_pro_time={}
    numonJobs=[]

    for i in range(n):
        line=f.readline()
        line_data=line.split()
        num_float = list(map(int, line_data))
        operation_num=int(num_float[0])
        numonJobs.append(operation_num)
        jj=1
        j=0
        while jj<len(num_float):
            o_num=int(num_float[jj])
            job_machines=[]
            job_processingtime=[]
            for kk in range(0,o_num*2,2):
                job_machines.append(num_float[jj+kk+1])
                job_processingtime.append(num_float[jj+kk+1+1])
            ops_mchs[(i+1,j+1)]=job_machines
            for l in range(len(job_machines)):
                ops_pro_time[(i + 1, j + 1,job_machines[l])] = job_processingtime[l]

            j+=1
            jj+=o_num*2+1
    f.close()  
    J=list(range(1,n+1)) 
    M=list(range(1,m+1)) 
    OJ={}
    for j in range(n):
        OJ[(J[j])]=list(range(1,numonJobs[j]+1))

    largeM=0
    for job in J:
        for op in OJ[(job)]:
            protimemax=0
            for l in ops_mchs[(job,op)]:
                if protimemax<ops_pro_time[(job,op,l)]:
                    protimemax=ops_pro_time[(job,op,l)]
            largeM+=protimemax
    Data={
        'n':n,
        'm':m,
        'J':J,
        'M':M,
        'OJ':OJ,
        'ops_mchs':ops_mchs,
        'ops_pro_time':ops_pro_time,
        'largeM':largeM,
    }
    return Data


def Validate(vali_set,batch_size, op_select_policy,mch_alloc_policy,op_num,num_task,Data):
    
    op_select_policy = copy.deepcopy(op_select_policy)
    mch_alloc_policy = copy.deepcopy(mch_alloc_policy)
    op_select_policy.eval()
    mch_alloc_policy.eval()
    def eval_model_bat(bat):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()
            env = FJSP(n_j=Data['n'], n_m=configs.n_m,Job_op=op_num)
            device = torch.device(configs.device)
            graph_pool_step = graph_pool(Type=configs.Type,
                                     batch_size=torch.Size(
                                    [batch_size, num_task, num_task]),
                                     n_nodes=num_task,
                                     device=device)

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)
            j = 0
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool=None
            while True:
                env_adj = Agr_ob(deepcopy(adj).to(device).to_sparse(), num_task)
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
                                                                                       , mch_pool=pool
                                                                                               ,old_policy=True,
                                                                                                T=1
                                                                                               ,greedy=True
                                                                                               )

                pi_mch,_,pool = mch_alloc_policy(action_node, hx, mask_mch_action, env_mch_time)

                _, mch_a = pi_mch.squeeze(-1).max(1)

                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a)
                if env.done():
                    break
            cost = env.mchsETs.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)
    
    all_time = torch.cat([eval_model_bat(bat) for bat in vali_set], 0)

    return all_time


def Get_path(path):
    return [os.path.join(path,f) for f in os.listdir(path)]


def test(filepath, datafile):

        Data = getdata(data_file)
        n_j = Data['n']
        n_m = Data['m']
        sdac = SDAC(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                    n_j=n_j,
                    n_m=n_m,
                    num_layers=configs.num_layers,
                    nighbor_pooling_type=configs.nighbor_pooling_type,
                    op_input_dim=configs.op_input_dim,
                    mch_input_dim=configs.mch_input_dim,
                    hidden_dim=configs.hidden_dim,
                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                    num_mlp_layers_actor=configs.num_mlp_layers_actor,
                    hidden_dim_actor=configs.hidden_dim_actor,
                    num_mlp_layers_critic=configs.num_mlp_layers_critic,
                    hidden_dim_critic=configs.hidden_dim_critic)

        job_path = './{}.pth'.format('op_select_policy')
        mch_path = './{}.pth'.format('mch_alloc_policy')

        job_path = os.path.join(filepath, job_path)
        mch_path = os.path.join(filepath, mch_path)

        sdac.op_select_policy.load_state_dict(torch.load(job_path))
        sdac.mch_alloc_policy.load_state_dict(torch.load(mch_path))

        num_val = 2
        batch_size = 2
        SEEDs = [200]
        op_nums = []
        op_num = []
        for i in Data['J']:
            op_num.append(Data['OJ'][i][-1])
        op_num_max = np.array(op_num).max()
        time_window = np.zeros(shape=(Data['n'], op_num_max, Data['m']))
        data_set = []
        for i in range(Data['n']):
            for j in Data['OJ'][i + 1]:
                mchForJob = Data['ops_mchs'][(i + 1, j)]
                for k in mchForJob:
                    time_window[i][j - 1][k - 1] = Data['ops_pro_time'][(i + 1, j, k)]
        for i in range(batch_size):
            op_nums.append(op_num)
            data_set.append(time_window)
        data_set = np.array(data_set)
        op_num = np.array(op_nums)
        num_tasks = op_num.sum(axis=1)[0]
        num_tasks = int(num_tasks)
        for SEED in SEEDs:
            mean_makespan = []
            np.random.seed(SEED)
            validat_dataset = Dataset(configs.n_j, configs.n_m, configs.low, configs.upper, num_val, SEED)
            valid_loader = DataLoader(data_set, batch_size=batch_size)
            vali_result = Validate(valid_loader, batch_size, sdac.op_select_policy, sdac.mch_alloc_policy, op_num,
                                   num_tasks, Data).mean()
            mean_makespan.append(vali_result)
            print(np.array(mean_makespan).mean())
        return np.array(mean_makespan).mean()


if __name__ == '__main__':

    filename = './pubilc_dataset/rdata-30-10'
    filename = Get_path(filename)
    print(filename)
    filepath = 'network'
    filepath = os.path.join(filepath, '10x10')
    filepaths = Get_path(filepath)
    print(filepaths)
    start_time = time.time()
    results = []
    for data_file in filename:
        result = []
        for filepath in filepaths:
            start_time_onefile = time.time()
            a = test(filepath, data_file)
            end_time_onefile = time.time()
            print(f"{filepath} {end_time_onefile - start_time_onefile}s")
            result.append(a)
        min = np.array(result).min()
        results.append(min)
    end_time = time.time()
    print(f"{end_time - start_time}s")
    print('mins', results)


