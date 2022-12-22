import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResLstm(nn.Module):
    def __init__(self, in_planes, stride=1):
        super(ResLstm, self).__init__()
        self.inplanes = in_planes
        self.inplanes_after1x1 = 3
        self.conv8_11 = nn.Sequential(

            nn.Conv1d(in_channels=self.inplanes_after1x1, out_channels=8, kernel_size=17, padding=8, stride=1),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=17, padding=8, stride=3),
        )
        self.conv16_9 = nn.Sequential(

            nn.Conv1d(in_channels=self.inplanes_after1x1, out_channels=8, kernel_size=13, padding=6, stride=1),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=13, padding=6, stride=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=13, padding=6, stride=3),

        )

        self.conv32_7 = nn.Sequential(

            nn.Conv1d(in_channels=self.inplanes_after1x1, out_channels=8, kernel_size=7, padding=3, stride=1),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, padding=3, stride=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3, stride=1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3, stride=3),

        )

        self.conv64_3 = nn.Sequential(

            nn.Conv1d(in_channels=self.inplanes_after1x1, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=3),

        )

        self.resconv64_3 =  self._make_layer(BasicBlock, 64, 2)
        self.resconv64_2 = self._make_layer(BasicBlock, 64, 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.bn_firstconv = nn.BatchNorm1d(3)
        self.tanh_firstconv = nn.ReLU()
        self.bn_afterconv = nn.BatchNorm1d(240)
        self.tanh_afterconv = nn.ReLU()
        self.bn_aftercat = nn.BatchNorm1d(131)
        self.tanh_aftercat = nn.ReLU()
        self.pool_afterconv = nn.MaxPool1d(kernel_size=3,stride=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.convfirst = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=3, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm1d(3),
            nn.ReLU(),
        )
        self.convfinal = nn.Sequential(
            nn.Conv1d(in_channels=240, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm1d(64, affine=True),
        )
        self.bilstm = nn.LSTM(input_size=1000, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True, bias=False)

        self.bn2 = nn.BatchNorm1d(16384)
        self.fc0 = nn.Linear(16384, 1, bias=False)


    def _make_layer(self, block, planes, blocks, stride=1):

        self.groups = 1
        self.base_width = 64
        layers = []
        layers.append(block(64, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(64, planes))

        return nn.Sequential(*layers)


    def forward(self, input, inputNorOri):

        inputNor = self.tanh_firstconv(self.bn_firstconv(self.convfirst(inputNorOri)))

        conv_input8_11 = self.conv8_11(inputNor)
        conv_input16_9 = self.conv16_9(inputNor)
        conv_input32_7 = self.conv32_7(inputNor)
        conv_input64_3 = self.conv64_3(inputNor)
        conv_ronghe = torch.cat((conv_input8_11,conv_input16_9), dim=1)
        conv_ronghe = torch.cat((conv_ronghe, conv_input32_7), dim=1)
        conv_ronghe = torch.cat((conv_ronghe, conv_input64_3), dim=1)
        conv_ronghe = self.convfinal(self.tanh_afterconv(self.bn_afterconv(conv_ronghe)))

        conv_input = self.resconv64_3(self.tanh_aftercat(conv_ronghe))
        conv_input = self.resconv64_2(conv_input)

        out,(fc,hc) = self.bilstm(self.bn1(conv_input))

        out = out.reshape(out.size(0), -1)
        out = self.fc0(self.drop(out))

        return out