#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : datasets.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/19 10:05

import torch
from torch.utils.data import Dataset

from data_process.get_data import InitMap2Data


class VehiclesData(Dataset):
    __constant__ = ['__grid_data', '__vehicles_id']

    def __init__(self, train=True):
        data = InitMap2Data()
        self.__grid_data, self.__vehicles_id, self.start = data.get_some_data()
        self.__train_grid, self.__train_vehicles = \
            self.__grid_data[:int(len(self.__grid_data) * 0.8)], self.__vehicles_id[:int(len(self.__grid_data) * 0.8)]
        self.__test_grid, self.__test_vehicles = \
            self.__grid_data[int(len(self.__grid_data) * 0.8):], self.__vehicles_id[int(len(self.__grid_data) * 0.8):]
        self.train = train

    def __getitem__(self, index):
        # s_position_lane, s_y, s_h, d, angle, s_lane, light, g_s, road_id, g_d
        # s_position_lane, s_h, d, angle, s_lane, light, g_s, road_id, g_d
        if self.train:
            data = torch.tensor(self.__train_grid[index], dtype=torch.float32)

            vehicle_id = self.__train_vehicles[index]  # 当前时间路网车辆id
        else:
            data = torch.tensor(self.__test_grid[index], dtype=torch.float32)
            vehicle_id = self.__test_vehicles[index]

        return data, vehicle_id, self.start + self.__vehicles_id.index(vehicle_id) + 1

    def __len__(self):
        return int(len(self.__grid_data) * 0.8) if self.train else int(len(self.__grid_data) * 0.2)
