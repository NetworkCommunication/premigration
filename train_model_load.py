import os
import scipy.io as sio
import torch
from torch import nn
import numpy as np
from model.gan_load import Generator, Discriminator

class TrainedModel(nn.Module):
    def __init__(self, learning_rate=0.0001, epochs=100):
        super().__init__()
        self.model_path='./assets/model/'

        self.generator_model='g_load.pth'
        self.discriminator_model='d_load.pth'
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        #print(os.path.isdir(self.model_path))
        self.generator_path = os.path.join(self.model_path, self.generator_model)
        self.discriminator_path = os.path.join(self.model_path, self.discriminator_model)
        self.train_data= sio.loadmat('./test_data/load_train.mat')['load_train']
        #print(self.train_data)
        self.test_data=sio.loadmat('./test_data/load_test.mat')['load_test']
        #print(self.test_data)


        if os.path.isfile(self.generator_path):  #如果有这个模型就加载，如果没有就新建
            self.generator = torch.load(self.generator_path)
        else:
            self.generator = Generator()

        if os.path.isfile(self.discriminator_path):
            self.discriminator = torch.load(self.discriminator_path)
        else:
            self.discriminator = Discriminator()

        self.generator.cuda()
        self.discriminator.cuda()


        self.g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate
        )
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate
        )





        #print(os.path.isfile(self.generator_path))
        self.milestones = [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000, 2500, 4000, 6000, 8000]
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.g_opt, milestones=self.milestones, gamma=0.5)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.d_opt, milestones=self.milestones, gamma=0.5)

        # gamma是倍数，每次乘以0.5，milestones是一个数组，epoch等于数组中的元素时，学习率乘一次倍数

        self.criterion = nn.BCELoss().cuda()

        self.criterion_mse = nn.MSELoss().cuda()


        self.device = torch.device("cuda")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.t_length = 2

    def train_model(self):
        for epoch in range(self.epochs):
            loss_g = 0
            loss_d = 0
            acc = 0
            acc_num = 0
            n = 0
            train_data=self.train_data
            n_t=len(train_data) /2
            n_t=int(n_t)
            #print(train_num)
            usernum_set = self.train_data[0:n_t-1]

            c_set = self.train_data[n_t:-1]
            #print(usernum_set)
            #print(c_set)
            for usernum,c in zip(usernum_set,c_set):
                #print(usernum)
                fake_label = torch.zeros(1).cuda()  # 全为0的数组
                real_label = torch.ones(1).cuda()  # 全为1的数组
                #print(fake_label)
                #print(real_label)

                usernum = torch.tensor(usernum,dtype=torch.float).cuda()
                real_o = torch.tensor(c,dtype=torch.float).cuda()

                fake_o = self.generator(usernum)
                d_real = self.discriminator(usernum, real_o.detach())  # 判别器对于真实数据的输出  应该接近于1  ，因为要判断多个车辆的数据，所以判别器的输出是个数组
                d_fake = self.discriminator(usernum, fake_o.detach())  # 判别器对于生成数据的输出  应该接近于0
                                                                    # real_o.detach() 返回一个没有梯度的新变量，这个新变量参与运算，在优化参数时real_o不参与
                self.d_opt.zero_grad()
                d_real_loss = self.criterion(d_real, real_label)  # criterion是计算损失函数用的
                d_fake_loss = self.criterion(d_fake, fake_label)
                d_loss = d_real_loss + d_fake_loss  # 判别器的损失函数
                d_loss.backward()  # 计算判别器的梯度
                self.d_opt.step()

                loss_d += d_loss

                self.g_opt.zero_grad()
                fake_o = self.generator(usernum)
                d_fake_1 = self.discriminator(usernum, fake_o)   #判别器对  虚假数据 的判断
                adversarial_loss = self.criterion(d_fake_1, real_label)
                mse_loss = self.criterion_mse(real_o, fake_o)
                g_loss = adversarial_loss + mse_loss  # 生成器的损失函数
                g_loss.backward()
                self.g_opt.step()



                loss_g += g_loss


            self.scheduler_d.step()
            self.scheduler_g.step()

            print(epoch)


            if epoch % 20 == 0 and epoch > 0:
                torch.save(self.generator, self.generator_path)
                torch.save(self.discriminator, self.discriminator_path)







        return



train_model=TrainedModel()
train_model.train_model()