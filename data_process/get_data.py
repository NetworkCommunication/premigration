#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : get_data.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/19 9:26

import json
import os
from config import *
from tools import get_lane_number

max_length = 5 * 100

config = Config()


def get_data_from_map(vehicles):
    vehicles_id = []
    vehicles = vehicles['vehicles']
    data_now = []
    for road_name in vehicles:
        vehicles_in_road = vehicles[road_name]['vehicles']

        if road_name in config.selected_road:
            for vid in vehicles_in_road:
                vehicle = vehicles_in_road[vid]
                position_lane = vehicle['info']['position_in_lane']
                road_id = config.selected_road.index(road_name)

                vehicles_id.append(vid)

                # position = vehicle['info']['position']
                lane_number = get_lane_number(vehicle['info']['lane'])
                # lateral_position = vehicle['info']['lateral_position']
                speed = vehicle['info']['speed']
                # v_length = vehicle['info']['length']

                s_num = 0
                min_distance = 500
                s_h = 0
                for sid in vehicles_in_road:
                    if sid != vid:
                        s_vehicle = vehicles_in_road[sid]
                        s_lane = get_lane_number(s_vehicle['info']['lane'])
                        s_position_lane = s_vehicle['info']['position_in_lane']
                        if s_lane == lane_number and 0 < (s_position_lane - position_lane) < max_length:
                            if min_distance > s_position_lane - position_lane:
                                s_num = 1
                                min_distance = s_position_lane - position_lane
                                s_h = s_vehicle['info']['speed']  # 水平速度
                                break

                d = min_distance if s_num == 1 else -1
                s_s = s_h if s_num == 1 else 0
                g_d = 500 - position_lane

                data_now.append([
                    position_lane, speed, g_d, s_num, d, s_s,
                    road_id,
                    lane_number
                ])

    return data_now, vehicles_id


class InitMap2Data:
    __constants__ = ['__data_data', '__vehicles_id']

    def __init__(
            self,
            load=True
    ):
        self.load = load
        self.traffic_file = os.path.join(
            config.path.data_path,
            config.traffic_filename)
        self.map_file = os.path.join(
            config.path.data_path,
            config.roadMap_filename)
        self.vehicle_file = os.path.join(
            config.path.data_path,
            config.vehicle_filename)

        self.__data_data = None
        self.__vehicles_id = None

        self.__info_file = os.path.join(
            config.path.data_path,
            config.vehicle_info)

        if not self.load:
            self.map, self.vehicles, self.traffic = self.__initialize_data()
        self.__get_current_data()

    def __initialize_data(self):
        with open(self.vehicle_file, 'r') as f:
            vehicles = json.load(f)

        with open(self.map_file, 'r') as f:
            maps = json.load(f)

        with open(self.traffic_file, 'r') as f:
            traffic = json.load(f)

        return maps, vehicles, traffic

    def __get_current_data(self):
        if not self.load:
            length = len(self.vehicles)
            data = []
            vehicles_id = []
            for i in range(length):
                if i % 20 == 0:
                    print("当前已经加载{}s的数据".format(i))
                data_now, vehicle_id = get_data_from_map(self.vehicles[i])
                data.append(data_now)
                vehicles_id.append(vehicle_id)

            self.__data_data = data
            self.__vehicles_id = vehicles_id
            self.__save()
        else:
            with open(self.__info_file, 'r') as f:
                res = json.loads(json.load(f))
                self.__data_data, self.__vehicles_id = res['data'], res['vid']

    def __save(self):
        res = json.dumps({
            'data': self.__data_data,
            'vid': self.__vehicles_id
        })
        with open(self.__info_file, 'w') as f:
            json.dump(res, f)

    def get_some_data(self):
        # position_lane, speed, g_d, s_num, d, s_s, light, road_id, lane_number, road_map.get_road_id(target)
        # position_lane, speed, g_d, s_num, d, s_s, light, road_id, lane_number, road_map.get_road_id(target)
        return self.__data_data[:-1], self.__vehicles_id[:-1], 0


if __name__ == '__main__':
    map2grip = InitMap2Data(load=False)
