#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : train_model.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/20 17:22

import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data_process.get_data import get_data_from_map
from data_process.vehicle_state import VehicleState
from datasets.datasets import VehiclesData
from model.gan import Generator, Discriminator

LENGTH = 500
MIN_DISTANCE = 5
min_loss = 100


def dataloader(batch_size=1, train=True): #加载数据。返回加载的数据
    label = '训练' if train else '测试'
    print("====================================加载{}数据====================================".format(label))
    grip_data = VehiclesData(train=train)
    loader = DataLoader(grip_data, batch_size=batch_size)
    print("加载完毕，{}数据数量为{}".format(label, len(grip_data)))
    return loader  #返回加载的数据


def del_tensor(tensors: torch.tensor, index: np.ndarray):  #删除tensor的梯度和 某一行的数据    tensors: torch.tensor  变量名：变量类型
    ten = tensors.detach().cpu().numpy()  #把tensor变量转化为数组
    ten = np.delete(ten, index, axis=0)   #按行删除  ndarray 数组，是用 np.ndarray类的对象 表示n维数组对象
    return torch.tensor(ten).cuda()    #再将ten这变量转换成tensor


def update_intersection_vehicle(intersection_vehicle: dict, cache=None):
    vids = []
    out_inter = {}
    need_del = []
    for k in intersection_vehicle:
        intersection_vehicle[k]['t'] -= 1

        if intersection_vehicle[k]['t'] == 0:
            out_inter[k] = intersection_vehicle[k]
            need_del.append(k)
        else:
            vids.append(k)

    for k in need_del:
        intersection_vehicle.pop(k)

    return intersection_vehicle, out_inter


def find_near_intersection_vehicle(state: list, vid: list):
    """
    :param state: [car_num, 3]  [lane_number, position, speed]
    :param vid: [car_num]
    :return:
    """

    near_list = list(filter(
        lambda v: (state[vid.index(v)][2] != 0 and LENGTH - state[vid.index(v)][1] - state[vid.index(v)][
            2] <= MIN_DISTANCE) or (
                          state[vid.index(v)][2] <= 0 and LENGTH - state[vid.index(v)][1] <= MIN_DISTANCE), vid))
    return near_list


class TrainedModel(nn.Module):
    def __init__(self, learning_rate=0.0001, epochs=40000):
        super().__init__()
        self.config = Config()
        if not os.path.isdir(self.config.path.model_path):
            os.mkdir(self.config.path.model_path)
        self.generator_path = os.path.join(self.config.path.model_path, self.config.generator_model)
        self.discriminator_path = os.path.join(self.config.path.model_path, self.config.discriminator_model)
        self.classifier_path = os.path.join(self.config.path.model_path, self.config.classifier_model)
        if os.path.isfile(self.generator_path):  #如果有这个模型就加载，如果没有就新建
            self.generator = torch.load(self.generator_path)
        else:
            self.generator = Generator()

        if os.path.isfile(self.classifier_path):
            self.classifier = torch.load(self.classifier_path)
        else:
            self.classifier = Generator(is_g=False)

        if os.path.isfile(self.discriminator_path):
            self.discriminator = torch.load(self.discriminator_path)
        else:
            self.discriminator = Discriminator()

        self.generator.cuda()
        self.classifier.cuda()
        self.discriminator.cuda()

        self.g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate
        )
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate
        )
        self.c_opt = torch.optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate
        )

        self.milestones = [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000, 2500, 4000, 6000, 8000]
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.g_opt, milestones=self.milestones, gamma=0.5)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.d_opt, milestones=self.milestones, gamma=0.5)
        self.scheduler_c = torch.optim.lr_scheduler.MultiStepLR(self.c_opt, milestones=self.milestones, gamma=0.5)
                    #gamma是倍数，每次乘以0.5，milestones是一个数组，epoch等于数组中的元素时，学习率乘一次倍数

        self.criterion = nn.BCELoss().cuda()
        self.criterion_classifier = nn.NLLLoss().cuda()
        self.criterion_mse = nn.MSELoss().cuda()

        self.vehicle_state = VehicleState()  # 获取车辆的状态真实值
        self.train_data = dataloader(train=True)
        self.test_data = dataloader(train=False)
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
            position_loss = 0
            speed_loss = 0
            n = 0
            for data in self.train_data:
                vs, vid, t_0 = data  # vs  :  [n_car, 10]  #data是个三元数组，所以用3个变量来存  vs, vid, t_0 过去的车辆速度 车辆id 时间
                vid = [v[0] for v in vid]
                vs = vs[0]
                vs = vs.cuda()

                real = self.vehicle_state.get_some_vehicles_state(int(t_0), vid)
                real = np.array(real)  #车辆状态

                error_list = np.where(real[:, 2] < 0)   #状态异常车辆的列表
                real = np.delete(real, error_list, axis=0)  #把异常的车辆状态删除
                vs = del_tensor(vs, error_list)    #把vs里面相应的行也删除，让vs和real两个数组形状相同

                position = np.zeros((vs.shape[0], 3))
                position_real = np.zeros((vs.shape[0], 3))

                fake_label = torch.zeros((vs.shape[0], 1)).cuda()  #全为0的数组
                real_label = torch.ones((vs.shape[0], 1)).cuda()   #全为1的数组

                fake_o = self.generator(vs)  # [n_car, 2]
                fake_c = self.classifier(vs)

                fake_t = torch.zeros((fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
                fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o

                real_t = np.zeros((fake_t.shape[0], fake_t.shape[1]))  #创建真实数据的集合，刚开始元素全为1
                real_t[:, :] = real[:, 3:]  # x, y         #赋值
                real_t = torch.tensor(real_t, dtype=torch.float32).cuda()  #转化为tensor变量

                real_c = real_t.detach().cpu().numpy()[:, 0]     #真实数据的第一列
                real_c = torch.tensor(real_c).cuda()

                real_o = real_t.detach().cpu().numpy()[:, 1:]
                real_o = torch.tensor(real_o).cuda()
                self.d_opt.zero_grad()                              #判别器的初始化
                d_real = self.discriminator(vs, real_o.detach())   #判别器对于真实数据的输出  应该接近于1  ，因为要判断多个车辆的数据，所以判别器的输出是个数组
                d_fake = self.discriminator(vs, fake_o.detach())    #判别器对于生成数据的输出  应该接近于0
                                                                    #real_o.detach() 返回一个没有梯度的新变量，这个新变量参与运算，在优化参数时real_o不参与

                d_real_loss = self.criterion(d_real, real_label)  #criterion是计算损失函数用的
                d_fake_loss = self.criterion(d_fake, fake_label)

                d_loss = d_real_loss + d_fake_loss     #判别器的损失函数
                d_loss.backward()                   #计算判别器的梯度
                self.d_opt.step()

                position[:, 2] = real[:, 2]  #空格 ： 2  ，等于从0到1，不包括2
                position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]
                position_real[:, 2] = real[:, 2]
                position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

                position_loss += np.mean(
                    np.sqrt(
                        ((position[:, 0] - position_real[:, 0]) * 3.2) ** 2 +
                        (position[:, 1] - position_real[:, 1]) ** 2
                    )
                )
                speed = fake_o.detach().cpu().numpy()[:, -1]   #-1是最后一个元素
                speed_real = real_t.detach().cpu().numpy()[:, -1]
                speed_loss += np.mean(
                    np.abs(speed - speed_real)
                )

                self.g_opt.zero_grad()
                fake_o = self.generator(vs)
                d_fake_1 = self.discriminator(vs, fake_o)
                adversarial_loss = self.criterion(d_fake_1, real_label)
                mse_loss = self.criterion_mse(real_o, fake_o)
                g_loss = adversarial_loss + mse_loss    #生成器的损失函数
                g_loss.backward()
                self.g_opt.step()

                loss_g += g_loss
                self.c_opt.zero_grad()
                nl_loss = self.criterion_classifier(fake_c, real_c.long())      #分类器的损失函数
                nl_loss.backward()
                self.c_opt.step()

                acc += torch.sum(fake_c.argmax(1) == real_c)
                acc_num += fake_c.shape[0]
                n += 1
                loss_d += d_loss

            print("epoch {} -------------------".format(epoch))
            print("classifier accuracy is {}".format(acc / acc_num))
            print("the average loss is {}".format(position_loss / n))
            print("loss_d is {}".format(loss_d / n))
            print("loss_g is {}".format(loss_g / n))
            print("speed loss is {}".format(speed_loss / n))

            self.scheduler_d.step()
            self.scheduler_g.step()
            self.scheduler_c.step()
            self.test_model()

            # if epoch >= 0:
            #     self.multi_test()

            if epoch % 20 == 0 and epoch > 0:
                torch.save(self.generator, self.generator_path)
                torch.save(self.discriminator, self.discriminator_path)
                torch.save(self.classifier, self.classifier_path)

    def multi_test(self):
        """
        :return:
        """
        length = 10
        classifier = self.classifier
        generator = self.generator
        num = [0 for _ in range(length)]
        acc = [0 for _ in range(length)]
        speed_loss = [0 for _ in range(length)]
        position_loss = [0 for _ in range(length)]   #数组的元素全是0，元素的数量是length

        n = 0

        for data in self.test_data:
            # s_y, s, d, angle, s_lane, light, g_s, road_id, g_d
            n += 1
            if n != 5:
                continue
            vs, vid, t_0 = data
            vs = vs[0]
            vid = [v[0] for v in vid]   #迭代过程中的所有的v[0]都是vid的元素
            vid = np.array(vid)
            vehicles_in_intersection = {}
            initial_info = self.vehicle_state.get_some_vehicles_state(int(t_0) - 1, vid)
            now_state = np.array(initial_info)[:, 3:6]

            classifier.eval()
            generator.eval()
            with torch.no_grad():
                for i in range(length):
                    vs = vs.cuda()
                    _input = vs

                    near_list = find_near_intersection_vehicle(now_state, vid.tolist())

                    real = self.vehicle_state.get_some_vehicles_state(int(t_0) + i, vid.tolist())
                    real = np.array(real)
                    need_del = []

                    for val in near_list:
                        need_del.append(np.argwhere(vid == val)[0])

                    if len(need_del) != 0:
                        need_out_list = np.array(need_del)
                        vid = np.delete(vid, need_out_list)
                        _input = del_tensor(_input, need_out_list)
                        real = np.delete(real, need_out_list, axis=0)

                    fake_o = generator(_input)
                    fake_c = classifier(_input)

                    fake_t = torch.zeros((fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
                    fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o

                    real_t = real[:, 3:6]
                    real_t = torch.tensor(real_t, dtype=torch.float32).cuda()

                    real_o = real_t.detach().cpu().numpy()[:, 1:]
                    real_o = torch.tensor(real_o).cuda()

                    real_c = real_t.detach().cpu().numpy()[:, 0]
                    real_c = torch.tensor(real_c).cuda()

                    position = np.zeros((_input.shape[0], 3))
                    position_real = np.zeros((_input.shape[0], 3))

                    position[:, 2] = real[:, 2]
                    position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]
                    position_real[:, 2] = real[:, 2]
                    position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

                    # [lane_name, position_in_lane, road_id]
                    error_list = np.where(position_real[:, 2] < 0)[0]
                    for val in error_list:
                        if position[val, 1] > 400:
                            position_real[val, 1] = 500
                        else:
                            position_real[val, 1] = 0
                        position_real[val, 2] = position[val, 2]

                    # [n_car, 2]
                    position_loss[i] += np.mean(
                        np.sqrt(
                            ((position[:, 0] - position_real[:, 0]) * 3.2) ** 2 +
                            (position[:, 1] - position_real[:, 1]) ** 2
                        )
                    )

                    num[i] += 1

                    speed = fake_o.detach().cpu().numpy()[:, -1]
                    speed_real = real_t.detach().cpu().numpy()[:, -1]
                    speed_loss[i] += np.mean(
                        np.abs(speed - speed_real)
                    )

                    acc[i] += (torch.sum(fake_c.argmax(1) == real_c) / fake_c.shape[0]).view(-1).item()

                    vehicles_in_intersection, out_inter = update_intersection_vehicle(vehicles_in_intersection)
                    output = fake_t.detach().cpu().numpy()
                    real_fake = real

                    grip_now, vehicles_id = self.make_grid(output, vid, real_fake, int(t_0) + i)
                    vs = grip_now
                    vid = np.array(vehicles_id)
                    now_state = np.zeros((grip_now.shape[0], 3))
                    now_state[:, 1:] = np.array(grip_now)[:, :2]
                    now_state[:, 0] = np.array(grip_now)[:, -2]

        acc = [acc[k] / num[k] for k, _ in enumerate(acc)]
        position_loss = [position_loss[k] / num[k] for k, _ in enumerate(position_loss)]
        speed_loss = [speed_loss[k] / num[k] for k, _ in enumerate(speed_loss)]

        print("classifier accuracy is ", acc)
        print("the average loss is ", position_loss)
        print("speed loss is ", speed_loss)

    def test_model(self):
        acc = 0
        acc_num = 0
        position_loss = 0
        speed_loss = 0
        n = 0
        for data in self.test_data:
            # if n != 0:
            #     break
            vs, vid, t_0 = data  # vs  :  [n_car, 10]
            vs = vs[0]
            vs = vs.cuda()
            vid = [v[0] for v in vid]  #迭代过程中的所有的v[0]都是vid的元素
            real = self.vehicle_state.get_some_vehicles_state(int(t_0), vid)
            real = np.array(real)

            # print(_input.shape)
            error_list = np.where(real[:, 2] < 0)  # 不需要放入神经网络训练的车辆
            real = np.delete(real, error_list, axis=0)
            vs = del_tensor(vs, error_list)

            position = np.zeros((vs.shape[0], 3))
            position_real = np.zeros((vs.shape[0], 3))

            fake_o = self.generator(vs)  # [n_car, 2]
            fake_c = self.classifier(vs)

            fake_t = torch.zeros((fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
            fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o #fake_t时预测值

            real_t = np.zeros((fake_t.shape[0], fake_t.shape[1]))
            real_t[:, :] = real[:, 3:6]  # x, y     real_t是真实值
            real_t = torch.tensor(real_t, dtype=torch.float32).cuda()

            real_c = real_t.detach().cpu().numpy()[:, 0]
            real_c = torch.tensor(real_c).cuda()

            real_o = real_t.detach().cpu().numpy()[:, 1:]
            real_o = torch.tensor(real_o).cuda()

            position[:, 2] = real[:, 2]
            position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]  #
            position_real[:, 2] = real[:, 2]
            position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

            position_loss += np.mean(   #对所有元素求均值
                np.sqrt(
                    ((position[:, 0] - position_real[:, 0]) * 3.2) ** 2 +
                    (position[:, 1] - position_real[:, 1]) ** 2
                )
            )
            speed = fake_o.detach().cpu().numpy()[:, -1]
            speed_real = real_t.detach().cpu().numpy()[:, -1]
            speed_loss += np.mean(
                np.abs(speed - speed_real)
            )

            acc += torch.sum(fake_c.argmax(1) == real_c)
            acc_num += fake_c.shape[0]
            n += 1

            # print(np.mean(
            #     np.sqrt(
            #         ((position[:, 0] - position_real[:, 0]) * 3.2) ** 2 +
            #         (position[:, 1] - position_real[:, 1]) ** 2
            #     )
            # ))
        print("classifier accuracy is {}".format(acc / acc_num))
        # print("learning rate is {}".format(self.scheduler_g.get_last_lr()))
        print("the average loss is {}".format(position_loss / n))
        print("speed loss is {}".format(speed_loss / n))
        print()


if __name__ == '__main__':
    train_model = TrainedModel()
    train_model.train_model()
