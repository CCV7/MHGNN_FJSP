import gym
import numpy as np
from gym.utils import EzPickle
from Gen_instance import override
from Params import configs
from Acontrol import Acontrol
from copy import deepcopy
import torch
import time

def get_Nor_action(action, ODM):

    co_a = np.where(ODM == action)

    precd = ODM[co_a[0], co_a[1] - 1 if co_a[1].item() > 0 else co_a[1]].item()

    suc_t = ODM[co_a[0], co_a[1] + 1 if co_a[1].item() + 1 < ODM.shape[-1] else co_a[1]].item()
    succd = action if suc_t < 0 else suc_t

def No_last(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)

    xAxis = np.arange(arr.shape[0], dtype=np.int64)

    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet

def calETL(tp1, min,mean):

    x, y = No_last(tp1, 1, invalid_val=-1)
    min[np.where(tp1 != 0)] = 0
    mean[np.where(tp1 != 0)] = 0
    min[x, y] = tp1[x, y]
    mean[x, y] = tp1[x, y]
    tp20 = np.cumsum(min, axis=1)
    tp21 = np.cumsum(mean, axis=1)
    tp20[np.where(tp1 != 0)] = 0
    tp21[np.where(tp1 != 0)] = 0
    tp2=np.concatenate((tp20.reshape(tp20.shape[0],tp20.shape[1],1),tp21.reshape(tp20.shape[0],tp20.shape[1],1)),-1)
    ret = tp1.reshape(tp1.shape[0],tp1.shape[1],1)+tp2
    return ret
def CTL(tp1, min):

    x, y = No_last(tp1, 1, invalid_val=-1)
    min[np.where(tp1 != 0)] = 0
    min[x, y] = tp1[x, y]
    tp20 = np.cumsum(min, axis=1)
    tp20[np.where(tp1 != 0)] = 0
    ret = tp1+tp20
    
    return ret

def op_mch_min(mch_time,job_time, METS, num_mch, dur, tp, first_col,mask_last,done,mm):
    tp = np.copy(tp)
    for machine, j in zip(METS, range(num_mch)):
        if np.all(machine == -configs.upper):
            mch_time[j] = 0
        else:
            mch_time[j] = machine[np.where(machine >= 0)][-1]
    for job, j in zip(tp, range(tp.shape[0])):
        if np.all(job == 0):
            job_time[j] = 0
        else:
            job_time[j] = job[np.where(job != 0)][-1]
    job_time1 = np.copy(job_time)
    while True:
        mask = np.full(shape=(tp.shape[0]), fill_value=1, dtype=bool)
        min_job_time = np.where(job_time1 <= job_time.min())[0]
        min_task = first_col[min_job_time]
        dur = deepcopy(dur)
        dur = dur.reshape(-1, num_mch)
        mchtask = []
        for z in min_task:
            mch_op = np.where(dur[z] > 0)[0]
            mchtask.append(mch_op)
        min_mchtask = []
        m_masks = np.copy(mm)

        for i in min_task:
            mask[np.where(first_col == i)] = 0
        mask = mask_last + mask
        if done:
            break
        elif np.all(mask) == True:
            job_time = np.delete(job_time, np.where(job_time == job_time.min())[0])
        else:
            break

    mch_space = min_mchtask
    Mch_space = min_task
    return mch_space, Mch_space,mask,m_masks

class FJSP(gym.Env, EzPickle):
    
    def __init__(self,
                 n_j,
                 n_m):
        EzPickle.__init__(self)
        self.step_count = 0
        self.num_job = n_j
        self.num_mch = n_m
        self.num_tasks = self.num_job * self.num_mch

        self.first_col = []
        self.last_col = []
        self.getETL = calETL
        self.getNghbs = get_Nor_action

    def done(self):
        if np.all(self.partial_sol_sequeence[0] >=0):
            return True
        return False

    @override
    def step(self, action,m_a,gantt_plt=None):

        time1 = time.time()
        feas, rewards, dones,masks,mch_masks = [],[], [], [],[]
        mch_spaces, mch_op_spaces = [],[]
        for i in range(self.batch_sie):
            if action[i] not in self.partial_sol_sequeence[i]:

                row = action[i] // self.num_mch
                col = action[i] % self.num_mch
                if i == 0:
                    self.step_count += 1
                self.finished_mark[i,row, col] = 1

                self.dur_a = self.dur[i,row, col,m_a[i]]
                self.partial_sol_sequeence[i][np.where(self.partial_sol_sequeence[i]<0)[0][0]] = action[i]

                self.m[i][row][col]=m_a[i]
                ST_a, flag = Acontrol(a=action[i], m_a=m_a[i], durMat=self.dur_cp[i], mchMat=self.m[i],
                                                         MST=self.MST[i], ODM=self.ODM[i],MET=self.METS[i])
                self.flags.append(flag)
                if gantt_plt != None:
                    gantt_plt.gantt_plt(row, col, m_a.cpu().numpy(), ST_a, self.dur_a,
                                    self.num_job)
                if action[i] not in self.last_col[i]:
                    self.omega[i,action[i] // self.num_mch] += 1
                else:
                    self.mask[i,action[i] // self.num_mch] = 1

                self.tp1[i,row, col] = ST_a + self.dur_a
                self.LBs[i] = calETL(self.tp1[i], self.input_min[i],self.input_mean[i])

                self.LBm[i] = CTL(self.tp1[i],self.input_min[i])

                precd, succd = self.getNghbs(action[i], self.ODM[i])

                self.adj[i, action[i]] = 0
                self.adj[i, action[i], action[i]] = 1
                if action[i] not in self.first_col[i]:
                    self.adj[i, action[i], action[i] - 1] = 1
                self.adj[i, action[i], precd] = 1
                self.adj[i, succd, action[i]] = 1

                done = self.done()
                mch_space,mch_op_space,mask1,mch_mask = op_mch_min(self.mch_time[i],self.job_time[i],self.METS[i],self.num_mch,self.dur_cp[i],self.tp1[i],self.omega[i],self.mask[i],done,self.mm[i])

                mch_spaces.append(mch_space)
                mch_op_spaces.append(mch_op_space)
                masks.append(mask1)
                mch_masks.append(mch_mask)
                print('action_space',mch_op_spaces,'mchspace',mch_space)

            fea = np.concatenate((self.LBm[i].reshape(-1, 1) / configs.nor_coef1,                               
                                  self.finished_mark[i].reshape( -1, 1)), axis=-1)
            feas.append(fea)

            reward = -(self.LBm[i].max() - self.max_endTime[i])
            if reward == 0:
                reward = configs.Reward_s
                self.posRewards[i] += reward
            rewards.append(reward)
            self.max_endTime[i] = self.LBm[i].max()

            dones.append(done)
        t2 = time.time()
        mch_masks = np.array(mch_masks)
        print(t2)
        return self.adj, np.array(feas), rewards, dones, self.omega, masks,mch_op_spaces,self.mm,self.mch_time,self.job_time

    @override
    def reset(self, data):
        
        self.batch_sie = data.shape[0]
        for i in range(self.batch_sie):
            first_col = np.arange(start=0, stop=self.num_tasks, step=1).reshape(self.num_job, -1)[:, 0]
            self.first_col.append(first_col)
        # 任务ID计算
            last_col = np.arange(start=0, stop=self.num_tasks, step=1).reshape(self.num_job, -1)[:, -1]
            self.last_col.append(last_col)
        self.first_col = np.array(self.first_col)
        self.last_col = np.array(self.last_col)
        self.step_count = 0
        self.m = -1 * np.ones((self.batch_sie,self.num_job,self.num_mch), dtype=np.int)
        self.dur = data.astype(np.single)
        self.dur_cp = deepcopy(self.dur)
        # 动作回收
        self.partial_sol_sequeence = -1 * np.ones((self.batch_sie,self.num_job*self.num_mch),dtype=np.int)
        self.flags = []
        self.posRewards = np.zeros(self.batch_sie)
        self.adj = []
        for i in range(self.batch_sie):
            op_ni_up_stream = np.eye(self.num_tasks, k=-1, dtype=np.single)
            op_ni_low_stream = np.eye(self.num_tasks, k=1, dtype=np.single)
            op_ni_up_stream[self.first_col] = 0
            op_ni_low_stream[self.last_col] = 0
            self_as_ni = np.eye(self.num_tasks, dtype=np.single)
            adj = self_as_ni + op_ni_up_stream
            self.adj.append(adj)
        self.adj = torch.tensor(self.adj)
        self.mm = np.full(shape=(self.batch_sie, self.num_job,self.num_mch, self.num_mch), fill_value=0,
                            dtype=bool)
        input_min=[]
        input_mean=[]
        start = time.time()
        for t in range(self.batch_sie):
            min = []
            mean = []
            for i in range(self.num_job):
                dur_min = []
                dur_mean = []
                for j in range(self.num_mch):
                    durmch = self.dur[t][i][j][np.where(self.dur[t][i][j] > 0)]

                    self.mm[t][i][j] = [1 if i <= 0 else 0 for i in self.dur_cp[t][i][j]]
                    self.dur[t][i][j] = [durmch.mean() if i <= 0 else i for i in self.dur[t][i][j]]
                    dur_min.append(durmch.min().tolist())
                    dur_mean.append(durmch.mean().tolist())
                min.append(dur_min)
                mean.append(dur_mean)
            input_min.append(min)
            input_mean.append(mean)
        end = time.time()-start

        self.input_min = np.array(input_min)
        self.input_mean =  np.array(input_mean)
        self.input_2d = np.concatenate([self.input_min.reshape((self.batch_sie,self.num_job,self.num_mch,1)),
                                        self.input_mean.reshape((self.batch_sie,self.num_job,self.num_mch,1))],-1)

        self.LBs = np.cumsum(self.input_2d,-2)
        self.LBm = np.cumsum(self.input_min,-1)

        self.initQuality = np.ones(self.batch_sie)
        for i in range(self.batch_sie):
            self.initQuality[i] = self.LBm[i].max() if not configs.init_flag else 0

        self.max_endTime = self.initQuality

        self.job_time = np.zeros((self.batch_sie, self.num_job))
        self.finished_mark = np.zeros_like(self.m)

        fea = np.concatenate((self.LBm.reshape(self.batch_sie,-1, 1) / configs.nor_coef1
                              ,self.finished_mark.reshape(self.batch_sie,-1, 1)), axis=-1)
        self.omega = self.first_col.astype(np.int64)
        self.mask = np.full(shape=(self.batch_sie,self.num_job), fill_value=0, dtype=bool)
        self.mch_time = np.zeros((self.batch_sie,self.num_mch))
        self.MST = -configs.upper * np.ones((self.batch_sie,self.num_mch,self.num_tasks))
        self.METS=-configs.upper * np.ones((self.batch_sie,self.num_mch,self.num_tasks))
        self.ODM = -self.num_job * np.ones((self.batch_sie,self.num_mch,self.num_tasks), dtype=np.int32)
        self.up_MET = np.zeros_like(self.METS)

        self.tp1 = np.zeros((self.batch_sie,self.num_job,self.num_mch))
        dur = self.dur.reshape(self.batch_sie,-1,self.num_mch)
        self.mm = self.mm.reshape(self.batch_sie,-1,self.mm.shape[-1])
        return self.adj, fea, self.omega, self.mask,self.mm,dur,self.mch_time,self.job_time