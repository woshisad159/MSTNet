from Train import train
from ReadConfig import readConfig
import random
import os
import numpy as np
import torch
import torch.backends.cudnn

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    # 读取配置文件
    configParams = readConfig()
    # isTrain为True是训练模式，isTrain为False是验证模式
    train(configParams, isTrain=True)

if __name__ == '__main__':
    seed_torch(10)
    main()

