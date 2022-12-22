#  coding:utf-8
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataset import MyDataset
from GJI.Submit import EMS_Model
from regular import Regularization

# Select Driver
CUDA_LAUNCH_BLOCKING="1"
device = torch.device("cuda:0")

# Tensorboard
Writer = SummaryWriter('runs/6-0-modify-norelu-256lstm-nozq')

# clusters: number of clusters, file: the file group experiment
clusters = 6
file = 0

# path of pkl
pkl = "Address.pkl"

# Loading Model
model = EMS_Model.ResLstm(3).cuda()


lr = 0.0001
lossFunction = torch.nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
reg_loss = Regularization(model, 0.1, p=2).to("cuda")

# Loading Dataset
print("train data")
trainloader = torch.utils.data.DataLoader(MyDataset(dataPath='TrainData Catalog Address',is_test = False), batch_size=64, shuffle=True)  # 训练集
print("test data")
trainloader_test = torch.utils.data.DataLoader(MyDataset(dataPath='TestData Catalog Address',is_test = True), batch_size=128, shuffle=True)  # 训练集



R2total = 0

totalTestAvg = 10000
count = 1000
losslog = []
avevalue = []

global_step_train = 0
global_step_test = 0


num = 1
def TestModel():
    global R2total, pkl

    model_test = torch.load(pkl)

    train_loss = 0  # accumulate every batch loss in a epoch
    outputs = 0
    targets = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs_Nor, target) in enumerate(trainloader_test):

            inputs, inputs_Nor, target = inputs.to(device), inputs_Nor.to(device), target.to(device)  # load data to gpu device
            output = model_test(inputs.type(torch.FloatTensor), inputs_Nor.type(torch.FloatTensor)) # test model

            output = output.squeeze(1)
            if batch_idx == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs,output),0)
                targets = torch.cat((targets,target),0)

        # Calculate mse
        targets = targets / 10
        outputs = outputs / 10
        loss = lossFunction(outputs, targets)  # compute loss
        train_loss += loss.item()  # accumulate every batch loss in a epoch

        # Calculate R2
        targets_mean = torch.mean(targets)
        outputs_cov = outputs - targets_mean
        targets_cov = targets - targets_mean
        outputs_r = 0
        targets_r = 0
        for num in range(len(outputs_cov)):
            outputs_r += outputs_cov[num] * outputs_cov[num]
            targets_r += targets_cov[num] * targets_cov[num]
        R2 = outputs_r / targets_r

        # Calculate std of outputs
        outputs_std = torch.std(torch.sub(targets-outputs))

        # Calculate mean
        avg = []
        avg.append(targets - outputs)
        avg = [j.detach().cpu().numpy() for j in avg]
        fin = np.mean(avg)

        print("Test loss: " + str((train_loss)) + " | R2 : " + str((R2)) +" | avg : " + str(fin) +"|std:"+str(outputs_std))

        if R2total < R2 :
            R2total = R2
            print("bestR2:" +str(R2total))


def TrainModel():
    global global_step_train
    for epoch in range(500):

        print('\nEpoch: %d' % epoch)
        model.train()  # enter train mode

        if epoch % 30 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8

        train_loss = 0  # accumulate every batch loss in a epoch
        Rs = 0
        total = []
        avg = []
        outputs_std = 0
        for batch_idx, (inputs, input_Nor, targets) in enumerate(trainloader):

            inputs, dataSet_Nor, targets = inputs.to(device), input_Nor.to(device), targets.to(device)  # load data to gpu device
            outputs = model(inputs.type(torch.FloatTensor), input_Nor.type(torch.FloatTensor))

            outputs = outputs.squeeze(1)
            optimizer.zero_grad()  # clear gradients of all optimized torch.Tensors'
            reg = reg_loss(model)
            loss = lossFunction(outputs, targets) + reg# compute loss
            loss.backward()  # compute gradient of loss over parameters
            optimizer.step()  # update parameters with gradient descent

            train_loss += loss.item()  # accumulate every batch loss in a epoch
            losslog.append(loss.item())

            targets = targets/10
            outputs = outputs/10
            targets_var = torch.var(targets)
            outputs_rmse = torch.mean(torch.pow(torch.sub(targets, outputs, alpha=1), 2))
            Rs += 1 - (outputs_rmse / targets_var).float().cpu()

            outputs_std += torch.std(outputs)

            avg1 = []
            avg1.append(targets - outputs)
            avg1 = [j.detach().cpu().numpy() for j in avg1]
            for i in range(len(avg1[0])):
                avg.append(avg1[0][i])

            if (batch_idx%10)==0:

                global_step_train += 1
                Writer.add_scalar(tag='6=0/train-loss',
                                  scalar_value=(train_loss / (batch_idx + 1)),
                                  global_step=global_step_train)

                Writer.add_scalar(tag='6=0/train-R2',
                                  scalar_value=(Rs / (batch_idx + 1)),
                                  global_step=global_step_train)

                fin = np.mean(avg)
                total.append(fin)
                a = str(np.max(avg))
                b = str(np.min(avg))
                print("Train loss: "+str((train_loss / (batch_idx + 1)))+" | avg: "+str(fin)+" | R2: "+str((Rs / (batch_idx + 1)))+"| rmse: "+str(np.mean(np.power(avg, 2)))+" ("+a+"/"+b+")"+"|std:"+str(outputs_std/(batch_idx + 1)))
                avg = []

        torch.save(model, pkl)



TrainModel()
TestModel()