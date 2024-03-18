import torch.nn as nn
import torch
import NewMoudle
import NewMoudle1

class moduleNet(nn.Module):
    def __init__(self, hiddenSize, wordSetNum, dataSetName='RWTH'):
        super().__init__()
        self.outDim = wordSetNum
        self.dataSetName = dataSetName
        self.logSoftMax = nn.LogSoftmax(dim=-1)

        self.conv2d = NewMoudle1.resnet18()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        hidden_size = hiddenSize
        inputSize = hiddenSize

        self.conv1D1_1 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=3, stride=1,
                                   padding=1)
        self.conv1D1_2 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=5, stride=1,
                                   padding=2)
        self.conv1D1_3 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=7, stride=1,
                                   padding=3)
        self.conv1D1_4 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=9, stride=1,
                                   padding=4)

        self.conv2D1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2,
                                padding=0)

        self.conv1D2_1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1,
                                   padding=1)
        self.conv1D2_2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, stride=1,
                                   padding=2)
        self.conv1D2_3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7, stride=1,
                                   padding=3)
        self.conv1D2_4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9, stride=1,
                                   padding=4)

        self.conv2D2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2,
                                 padding=0)

        self.batchNorm1d1_1 = nn.BatchNorm1d(hidden_size)
        self.batchNorm1d1_2 = nn.BatchNorm1d(hidden_size)
        self.batchNorm1d1_3 = nn.BatchNorm1d(hidden_size)
        self.batchNorm1d1_4 = nn.BatchNorm1d(hidden_size)

        self.batchNorm1d2_1 = nn.BatchNorm1d(hidden_size)
        self.batchNorm1d2_2 = nn.BatchNorm1d(hidden_size)
        self.batchNorm1d2_3 = nn.BatchNorm1d(hidden_size)
        self.batchNorm1d2_4 = nn.BatchNorm1d(hidden_size)

        self.batchNorm2d1 = nn.BatchNorm2d(hidden_size)
        self.batchNorm2d2 = nn.BatchNorm2d(hidden_size)

        self.relu = nn.ReLU(inplace=True)

        heads = 8
        semantic_layers = 2
        dropout = 0
        rpe_k = 8
        self.temporal_model = NewMoudle.TransformerEncoder(hidden_size, heads, semantic_layers, dropout, rpe_k)

        self.linear1 = nn.Linear(512, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.batchNorm1d1 = nn.BatchNorm1d(hidden_size)
        self.batchNorm1d2 = nn.BatchNorm1d(hidden_size)

        self.classifier1 = nn.Linear(hidden_size, self.outDim)
        self.classifier2 = nn.Linear(hidden_size, self.outDim)

        if self.dataSetName == 'RWTH':
            self.classifier3 = nn.Linear(hidden_size, self.outDim)
            self.classifier4 = nn.Linear(inputSize, self.outDim)


    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    def forward(self, seqData, dataLen, isTrain=True):
        len_x = dataLen
        batch, temp, channel, height, width = seqData.shape
        x = seqData.transpose(1, 2)
        framewise = self.conv2d(x)

        framewise = framewise.reshape(batch, temp, -1)
        ######################################
        framewise = self.linear1(framewise).transpose(1, 2)
        framewise = self.batchNorm1d1(framewise)
        framewise = self.relu(framewise).transpose(1, 2)
        #
        framewise = self.linear2(framewise).transpose(1, 2)
        framewise = self.batchNorm1d2(framewise)
        framewise = self.relu(framewise)
        ######################################
        inputData = self.conv1D1_1(framewise)
        inputData = self.batchNorm1d1_1(inputData)
        inputData = self.relu(inputData)

        glossCandidate = inputData.unsqueeze(2)

        inputData = self.conv1D1_2(framewise)
        inputData = self.batchNorm1d1_2(inputData)
        inputData = self.relu(inputData)

        tmpData = inputData.unsqueeze(2)
        glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

        inputData = self.conv1D1_3(framewise)
        inputData = self.batchNorm1d1_3(inputData)
        inputData = self.relu(inputData)

        tmpData = inputData.unsqueeze(2)
        glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

        inputData = self.conv1D1_4(framewise)
        inputData = self.batchNorm1d1_4(inputData)
        inputData = self.relu(inputData)

        tmpData = inputData.unsqueeze(2)
        glossCandidate = torch.cat([glossCandidate, tmpData], dim = 2)

        inputData = self.conv2D1(glossCandidate)
        inputData = self.batchNorm2d1(inputData)
        inputData1 = self.relu(inputData).squeeze(2)
        ######################################
        # 2
        inputData = self.conv1D2_1(inputData1)
        inputData = self.batchNorm1d2_1(inputData)
        inputData = self.relu(inputData)

        glossCandidate = inputData.unsqueeze(2)

        inputData = self.conv1D2_2(inputData1)
        inputData = self.batchNorm1d2_2(inputData)
        inputData = self.relu(inputData)

        tmpData = inputData.unsqueeze(2)
        glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

        inputData = self.conv1D2_3(inputData1)
        inputData = self.batchNorm1d2_3(inputData)
        inputData = self.relu(inputData)

        tmpData = inputData.unsqueeze(2)
        glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

        inputData = self.conv1D2_4(inputData1)
        inputData = self.batchNorm1d2_4(inputData)
        inputData = self.relu(inputData)

        tmpData = inputData.unsqueeze(2)
        glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

        inputData = self.conv2D2(glossCandidate)
        inputData = self.batchNorm2d2(inputData)
        inputData = self.relu(inputData).squeeze(2)
        ######################################
        if self.dataSetName == 'RWTH':
            lgt = torch.cat(len_x, dim=0) // 4
            x = inputData.permute(0, 2, 1)
        else:
            lgt = (torch.cat(len_x, dim=0) // 4) - 2
            x = inputData.permute(0, 2, 1)
            x = x[:,1:-1,:]

        outputs = self.temporal_model(x)

        outputs = outputs.permute(1, 0, 2)
        encoderPrediction = self.classifier1(outputs)
        logProbs1 = self.logSoftMax(encoderPrediction)

        outputs = x.permute(1, 0, 2)
        encoderPrediction = self.classifier2(outputs)
        logProbs2 = self.logSoftMax(encoderPrediction)

        if self.dataSetName == 'RWTH':
            outputs = inputData1.permute(2, 0, 1)
            encoderPrediction = self.classifier3(outputs)
            logProbs3 = self.logSoftMax(encoderPrediction)

            outputs = framewise.permute(2, 0, 1)
            encoderPrediction = self.classifier4(outputs)
            logProbs4 = self.logSoftMax(encoderPrediction)
        else:
            logProbs3 = 0
            logProbs4 = 0

        return logProbs1, logProbs2, logProbs3, logProbs4, lgt

