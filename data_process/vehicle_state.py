#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : vehicle_state.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/19 9:26

import json
import os

from config import *
from tools import get_lane_number


class VehicleState:
    def __init__(self, load=True):
        self.config = Config()
        self.vehicle_file = os.path.join(
            self.config.path.data_path,
            self.config.vehicle_filename
        )
        self.vehicle_state_file = os.path.join(
            self.config.path.data_path,
            self.config.vehicle_state
        )
        self.load = load
        self.vehicle_state = None
        self.vehicle_trip = None
        if not load:
            self.vehicles = self.__initialize_data()
        self.__get_current_data()

    def __initialize_data(self):
        with open(self.vehicle_file, 'r') as f:
            vehicles = json.load(f)

        return vehicles

    def __get_current_data(self):
        if not self.load:
            v_state = []
            for v in self.vehicles:
                vehicles = v['vehicles']
                s_per = {}
                for road_name in vehicles:  # 获取每条道路中的车辆
                    # 如果道路存在id则获取id，如果是交叉路口那么直接使用交叉路口名
                    if road_name not in self.config.selected_road:
                        continue
                    road_id = self.config.selected_road.index(road_name)
                    r = vehicles[road_name]['vehicles']
                    for vid in r:  # 获取每个车辆的信息
                        vehicle = r[vid]
                        x, y = vehicle['info']['position']
                        position_in_lane = vehicle['info']['position_in_lane']
                        v_s = vehicle['info']['speed']
                        # h_s = vehicle['info']['lateral_speed']
                        lane_num = get_lane_number(vehicle['info']['lane'])
                        # a = vehicle['info']['accelerate']
                        s_per[vid] = [x, y, road_id, lane_num, round(position_in_lane, 2), round(v_s, 2),
                                      # round(a, 2),
                                      # round(h_s, 2)
                                      ]
                v_state.append(s_per)
            self.vehicle_state = v_state
            with open(self.vehicle_state_file, 'w') as f:
                json.dump(json.dumps(v_state), f)
        else:
            with open(self.vehicle_state_file, 'r') as f:
                v_state = json.loads(json.load(f))
                self.vehicle_state = v_state

    def get_one_vehicle_state(self, t: int, vid: str) -> list:
        return self.vehicle_state[t][vid]

    def get_some_vehicles_state(self, t: int, vids: list) -> list:
        """
        get some vehicles' state
        :param t: time
        :param vids: vehicles id list
        :return states: vehicles state
        """
        states = []
        for vid in vids:
            try:  # 如果找不到目标的话，说明该车已经到达的目的地，那么返回0
                # 如果车辆进入交叉路口，那么返回进入交叉路口前的位址, 道路编号置为-1
                res = self.get_one_vehicle_state(t - 1, vid) if type(
                    self.get_one_vehicle_state(t, vid)[2]) == str else self.vehicle_state[t][vid]
                res[2] = res[2] if type(
                    self.get_one_vehicle_state(t, vid)[2]) == int else -1
                res[4] = res[4] if type(
                    self.get_one_vehicle_state(t, vid)[2]) != -1 else 500
                states.append(res)
            except KeyError:
                res = [0, 0, -2, 0, 500, 0]
                states.append(res)
        return states

    def get_some_vehicle_position(self, t: int, vids: list):
        states = []
        for vid in vids:
            try:
                res = self.get_one_vehicle_state(t - 1, vid) if type(
                    self.get_one_vehicle_state(t, vid)[2]) == str else self.vehicle_state[t][vid]
                res[2] = res[2] if type(
                    self.get_one_vehicle_state(t, vid)[2]) == int else -1
                res[4] = res[4] if type(
                    self.get_one_vehicle_state(t, vid)[2]) != -1 else 500
                states.append(res)
            except KeyError:
                res = [0, 0, -2, 0, 500, 0, 0, 0]
                # res = self.vehicle_state[t - 1][vid]
                # res[2] = -2
                states.append(res)


if __name__ == '__main__':
    vehicle_state = VehicleState(load=False)
    print(vehicle_state.get_one_vehicle_state(10, '495'))
    # print(vehicle_state.get_vehicle_trip('62'))
