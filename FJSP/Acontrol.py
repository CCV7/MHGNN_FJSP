from Params import configs
import numpy as np

def Acontrol(a,m_a, durMat, mchMat, MST, ODM,MET):

    JRT_a, MRT_a = CMRT(a,m_a, mchMat, durMat, MST, ODM)
    dur_a = durMat[a//durMat.shape[1]][a%durMat.shape[1]][m_a]
    STMF = MST[m_a]
    ETFM=MET[m_a]
    ODFM = ODM[m_a]
    flag = False
    POS_P = np.where(JRT_a < STMF)[0]

    if len(POS_P) == 0:
        ST_a = P_End(a, JRT_a, MRT_a, STMF, ODFM,ETFM,dur_a)
    else:
        idxLPos, LPos, ETPOS_P = Pos(dur_a,m_a, JRT_a, durMat, POS_P, STMF, ODFM)
        if len(LPos) == 0:
            ST_a = P_End(a, JRT_a, MRT_a, STMF, ODFM,ETFM,dur_a)
        else:
            flag = True
            ST_a = P_Bet(a, idxLPos, LPos, ETPOS_P, STMF, ODFM,ETFM,dur_a)
    return ST_a, flag

def P_End(a, JRT_a, MRT_a, STMF, ODFM,ETFM,dur_a):

    index = np.where(STMF == -configs.upper)[0][0]
    ST_a = max(JRT_a, MRT_a)
    STMF[index] = ST_a
    ODFM[index] = a
    ETFM[index]=ST_a+dur_a
    return ST_a


def Pos(dur_a,m_a,JRT_a, durMat, POS_P, STFMO, ODFM):

    STPOS_P = STFMO[POS_P]
    durOfPOS_P=[]
    for posPo in POS_P:
        durOfPOS_P.append(durMat[ODFM[posPo]//durMat.shape[1]][ODFM[posPo]% durMat.shape[1]][m_a])
    durOfPOS_P=np.array(durOfPOS_P)
    STest = max(JRT_a, STFMO[POS_P[0]-1] + durMat[ODFM[POS_P[0]-1]//durMat.shape[1]][ODFM[POS_P[0]-1]% durMat.shape[1]][m_a])
    ETPOS_P = np.append(STest, (STPOS_P + durOfPOS_P))[:-1]
    posGaps = STPOS_P - ETPOS_P
    idxLPos = np.where(dur_a <= posGaps)[0]
    LPos = np.take(POS_P, idxLPos)
    return idxLPos, LPos, ETPOS_P

def P_Bet(a, idxLPos, LPos, ETPOS_P, STFMO, ODFM,ETFM,dur_a):

    estIdx = idxLPos[0]
    estPos = LPos[0]
    ST_a = ETPOS_P[estIdx]
    STFMO[:] = np.insert(STFMO, estPos, ST_a)[:-1]
    ETFM[:]=np.insert(ETFM, estPos, ST_a+dur_a)[:-1]
    ODFM[:] = np.insert(ODFM, estPos, a)[:-1]
    return ST_a

def CMRT(a, m_a,mchMat, durMat, MST, ODM):

    op_pred = a - 1 if a % durMat.shape[1] != 0 else None
    if op_pred is not None:
        mchop_pred = np.take(mchMat, op_pred)
        durop_pred = durMat[op_pred//durMat.shape[1],op_pred%durMat.shape[1],mchop_pred]
        JRT_a = (MST[mchop_pred][np.where(ODM[mchop_pred] == op_pred)] + durop_pred).item()
    else:
        JRT_a = 0
    mch_pred = ODM[m_a][np.where(ODM[m_a] >= 0)][-1] if len(np.where(ODM[m_a] >= 0)[0]) != 0 else None
    if mch_pred is not None:
        durMch_pred = durMat[mch_pred//durMat.shape[1],mch_pred%durMat.shape[1],m_a]
        MRT_a = (MST[m_a][np.where(MST[m_a] >= 0)][-1] + durMch_pred).item()
    else:
        MRT_a = 0
    return JRT_a, MRT_a

if __name__ == "__main__":
    from Env import FJSP
    from Gen_instance import ins_gen,Dataset
    import time
    from torch.utils.data import DataLoader
    n_j = configs.n_j
    n_m = configs.n_m
    low = configs.low
    upper = configs.upper
    SEED = 200
    np.random.seed(SEED)
    t3 = time.time()
    train_dataset = Dataset(n_j, n_m, low, upper,2)
    data_loader = DataLoader(train_dataset, batch_size=2)
    for batch_idx, data_set in enumerate(data_loader):
        data_set = data_set.numpy()
        batch_size = data_set.shape[0]
        env = FJSP(n_j=n_j, n_m=n_m)
        t1 = time.time()
        MST = -configs.upper * np.ones((n_m,n_m*n_j), dtype=np.int32)
        mchsEndtTimes = -configs.upper * np.ones((n_m, n_m * n_j), dtype=np.int32)
        ODM = -n_j * np.ones([n_m,n_m*n_j], dtype=np.int32)
        adj, _, omega, mask,mch_mask,_,mch_time,_ = env.reset(data_set)
        print(adj)
        print(data_set)
        print(env.adj)
        mch_mask = mch_mask.reshape(batch_size, -1,n_m)
        job = omega
        rewards = []
        flags = []
        while True:
            action = []
            m_a = []
            for i in range(batch_size):
                a= np.random.choice(omega[i][np.where(mask[i] == 0)])
                m = np.random.choice(np.where(mch_mask[i][a] == 0)[0])
                action.append(a)
                m_a.append(m)
            adj, _, reward, done, omega, mask,job,_,mch_time,_= env.step(action,m_a)
            rewards.append(reward)
            if env.done():
                break
        t2 = time.time()
        print(t2 - t1)
        