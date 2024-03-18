import Moudle
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from WER import WerScore
import os
import DataProcessMoudle
import DecodeMoudle
import videoAugmentation
import numpy as np
import decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from evaluation import evaluteMode

def train(configParams, isTrain=True):
    # 参数初始化
    # 读入数据路径
    trainDataPath = configParams["trainDataPath"]
    validDataPath = configParams["validDataPath"]
    testDataPath = configParams["testDataPath"]
    # 读入标签路径
    trainLabelPath = configParams["trainLabelPath"]
    validLabelPath = configParams["validLabelPath"]
    testLabelPath = configParams["testLabelPath"]
    # 读入模型参数
    bestModuleSavePath = configParams["bestModuleSavePath"]
    currentModuleSavePath = configParams["currentModuleSavePath"]
    # 读入参数
    device = configParams["device"]
    hiddenSize = int(configParams["hiddenSize"])
    lr = float(configParams["lr"])
    batchSize = int(configParams["batchSize"])
    numWorkers = int(configParams["numWorkers"])
    pinmMemory = bool(int(configParams["pinmMemory"]))
    dataSetName = configParams["dataSetName"]
    max_num_states = 1
    sourcefilePath = './evaluation/wer/evalute'
    if isTrain:
        fileName = "output-hypothesis-{}.ctm".format('dev')
    else:
        fileName = "output-hypothesis-{}.ctm".format('test')
    filePath = os.path.join(sourcefilePath, fileName)

    # 预处理语言序列
    word2idx, wordSetNum, idx2word = DataProcessMoudle.Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName)
    # 图像预处理
    if dataSetName == "RWTH":
        transform = videoAugmentation.Compose([
            videoAugmentation.RandomCrop(224),
            videoAugmentation.RandomHorizontalFlip(0.5),
            videoAugmentation.ToTensor(),
        ])

        transformTest = videoAugmentation.Compose([
            videoAugmentation.CenterCrop(224),
            videoAugmentation.ToTensor(),
        ])
    elif dataSetName == "CSL":
        transform = videoAugmentation.Compose([
            videoAugmentation.RandomCrop(224),
            videoAugmentation.ToTensor(),
        ])

        transformTest = videoAugmentation.Compose([
            videoAugmentation.CenterCrop(224),
            videoAugmentation.ToTensor(),
        ])

    # 导入数据
    trainData = DataProcessMoudle.MyDataset(trainDataPath, trainLabelPath, word2idx, dataSetName, isTrain=True, transform=transform)

    validData = DataProcessMoudle.MyDataset(validDataPath, validLabelPath, word2idx, dataSetName, transform=transformTest)

    if dataSetName == "RWTH":
        testData = DataProcessMoudle.MyDataset(testDataPath, testLabelPath, word2idx, dataSetName, transform=transformTest)

    trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    validLoader = DataLoader(dataset=validData, batch_size=batchSize, shuffle=False, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    if dataSetName == "RWTH":
        testLoader = DataLoader(dataset=testData, batch_size=batchSize, shuffle=False, num_workers=numWorkers,
                                pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    else:
        testLoader = validLoader
    # 定义模型
    moduleNet = Moudle.moduleNet(hiddenSize, wordSetNum*max_num_states + 1, dataSetName).to(device)
    # 损失函数定义
    PAD_IDX = 0#wordSetNum * max_num_states

    ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='none', zero_infinity=False).to(device)
    logSoftMax = nn.LogSoftmax(dim=-1)
    kld = DataProcessMoudle.SeqKD(T=8).to(device)
    # 优化函数
    params = list(moduleNet.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0001)
    # 读取预训练模型参数
    bestLoss = 65535
    bestLossEpoch = 0
    bestWerScore = 65535
    bestWerScoreEpoch = 0
    epoch = 0
    if os.path.exists(currentModuleSavePath):
        checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'))
        moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        bestLoss = checkpoint['bestLoss']
        bestLossEpoch = checkpoint['bestLossEpoch']
        bestWerScore = checkpoint['bestWerScore']
        bestWerScoreEpoch = checkpoint['bestWerScoreEpoch']
        epoch = checkpoint['epoch']
        lastEpoch = epoch
        print(f"已加载预训练模型 epoch: {epoch}, bestLoss: {bestLoss:.5f}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")
    else:
        lastEpoch = -1
        print(f"未加载预训练模型 epoch: {epoch}, bestLoss: {bestLoss}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")
    # 设置学习率衰减规则
    if dataSetName == "CSL":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=[20],
                                                         gamma=0.1, last_epoch=lastEpoch)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=[45, 65],
                                                         gamma=0.2, last_epoch=lastEpoch)
    # 解码参数
    beam_width = 1
    prune = 0.01
    decodeMode = DecodeMoudle.Model(wordSetNum, max_num_states = max_num_states)

    decoder = decode.Decode(word2idx, wordSetNum + 1, 'beam')

    if isTrain:
        print("开始训练模型")
        # 训练模型
        if dataSetName == "CSL":
            epochNum = 30
        else:
            epochNum = 85

        for _ in range(epochNum):
            moduleNet.train()
            scaler = GradScaler()
            loss_value = []
            for Dict in tqdm(trainLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]

                targetOutData = [torch.tensor(decodeMode.decoder.expand(yi)).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetOutData = torch.cat(targetOutData, dim=0).to(device)

                with autocast():
                    logProbs1, logProbs2, logProbs3, logProbs4, lgt = moduleNet(data, dataLen, True)

                    if dataSetName == "RWTH":
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss3 = ctcLoss(logProbs3, targetOutData, lgt * 2, targetLengths)
                        loss4 = ctcLoss(logProbs4, targetOutData, lgt * 4, targetLengths)
                        loss = loss1 + loss2 + loss3 + loss4
                    elif dataSetName == "CSL":
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss = loss1 + loss2

                    optimizer.zero_grad()

                    # 对loss进行缩放，针对缩放后的loss进行反向传播
                    # （此部分计算在autocast()作用范围以外）
                    scaler.scale(loss).backward()

                    # 将梯度值缩放回原尺度后，优化器进行一步优化
                    scaler.step(optimizer)

                    # 更新scalar的缩放信息
                    scaler.update()

                loss_value.append(loss.item())

                torch.cuda.empty_cache()

            print("epoch: %d, trainLoss: %f, lr : %f" % (
            epoch, np.mean(loss_value), optimizer.param_groups[0]['lr']))

            epoch = epoch + 1

            scheduler.step()

            moduleNet.eval()
            print("开始验证模型")
            # 验证模型
            werScoreSum = 0
            loss_value = []
            total_info = []
            total_sent = []

            for Dict in tqdm(validLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]

                targetOutData = [torch.tensor(decodeMode.decoder.expand(yi)).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                batchSize = len(targetLengths)

                with torch.no_grad():
                    logProbs1, logProbs2, logProbs3, logProbs4, lgt = moduleNet(data, dataLen, False)

                    if dataSetName == "RWTH":
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss3 = ctcLoss(logProbs3, targetOutData, lgt * 2, targetLengths)
                        loss4 = ctcLoss(logProbs4, targetOutData, lgt * 4, targetLengths)
                        loss = loss1 + loss2 + loss3 + loss4
                    elif dataSetName == "CSL":
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss = loss1 + loss2

                loss_value.append(loss.item())

                if dataSetName == "RWTH":
                    pred = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                    total_info += info
                    total_sent += pred
                elif dataSetName == "CSL":
                    prob = []
                    P = logProbs1.permute(1, 0, 2)
                    prob += [lpi.exp().cpu().numpy() for lpi in P]
                    targetOutDataCTC = decodeMode.decode(prob, beam_width, prune)

                    werScore = WerScore(targetOutDataCTC, targetData, idx2word, batchSize)
                    werScoreSum = werScoreSum + werScore

                torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(validLoader)

            if werScore < bestWerScore:
                bestWerScore = werScore
                bestWerScoreEpoch = epoch - 1

                moduleDict = {}
                moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
                moduleDict['optimizer_state_dict'] = optimizer.state_dict()
                moduleDict['bestLoss'] = bestLoss
                moduleDict['bestLossEpoch'] = bestLossEpoch
                moduleDict['bestWerScore'] = bestWerScore
                moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
                moduleDict['epoch'] = epoch
                torch.save(moduleDict, bestModuleSavePath)

            bestLoss = currentLoss
            bestLossEpoch = epoch - 1

            moduleDict = {}
            moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
            moduleDict['optimizer_state_dict'] = optimizer.state_dict()
            moduleDict['bestLoss'] = bestLoss
            moduleDict['bestLossEpoch'] = bestLossEpoch
            moduleDict['bestWerScore'] = bestWerScore
            moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
            moduleDict['epoch'] = epoch
            torch.save(moduleDict, currentModuleSavePath)

            moduleSavePath1 = 'module/bestMoudleNet_' + str(epoch) + '.pth'
            torch.save(moduleDict, moduleSavePath1)

            if dataSetName == "RWTH":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)
                evaluteMode('evalute_dev')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('dev', epoch), total_info, total_sent)

            print(f"validLoss: {currentLoss:.5f}, werScore: {werScore:.5f}")
            print(f"bestLoss: {bestLoss:.5f}, beatEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.5f}, bestWerScoreEpoch: {bestWerScoreEpoch}")
    else:
        for i in range(80):
            currentModuleSavePath = "module/bestMoudleNet_" + str(i + 1) + ".pth"
            checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'))
            moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            bestLoss = checkpoint['bestLoss']
            bestLossEpoch = checkpoint['bestLossEpoch']
            bestWerScore = checkpoint['bestWerScore']
            bestWerScoreEpoch = checkpoint['bestWerScoreEpoch']

            moduleNet.eval()
            print("开始验证模型")
            # 验证模型
            werScoreSum = 0
            loss_value = []
            total_info = []
            total_sent = []

            for Dict in tqdm(testLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]

                targetOutData = [torch.tensor(decodeMode.decoder.expand(yi)).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                batchSize = len(targetLengths)

                with torch.no_grad():
                    logProbs1, logProbs2, logProbs3, logProbs4, lgt = moduleNet(data, dataLen, False)

                    if dataSetName == "RWTH":
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss3 = ctcLoss(logProbs3, targetOutData, lgt * 2, targetLengths)
                        loss4 = ctcLoss(logProbs4, targetOutData, lgt * 4, targetLengths)
                        loss = loss1 + loss2 + loss3 + loss4
                    elif dataSetName == "CSL":
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss = loss1 + loss2

                loss_value.append(loss.item())

                if dataSetName == "RWTH":
                    pred = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                    total_info += info
                    total_sent += pred
                elif dataSetName == "CSL":
                    prob = []
                    P = logProbs1.permute(1, 0, 2)
                    prob += [lpi.exp().cpu().numpy() for lpi in P]
                    targetOutDataCTC = decodeMode.decode(prob, beam_width, prune)

                    werScore = WerScore(targetOutDataCTC, targetData, idx2word, batchSize)
                    werScoreSum = werScoreSum + werScore

                torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(testLoader)

            if dataSetName == "RWTH":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)
                evaluteMode('evalute_test')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('test', i + 1), total_info,
                                             total_sent)

            print(f"validLoss: {currentLoss:.5f}, werScore: {werScore:.5f}")
            print(f"bestLoss: {bestLoss:.5f}, beatEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.5f}, bestWerScoreEpoch: {bestWerScoreEpoch}")


