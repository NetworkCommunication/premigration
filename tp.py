#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : tp.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/19 9:19
import os

from train_model import *
import numpy as np
from config import Config

LENGTH = 500
MIN_DISTANCE = 5
min_loss = 100


class TP(object):
    """
    trajectory prediction
    """

    def __init__(self):
        super().__init__()
        self.config = Config()

        self.generator_path = os.path.join(
            self.config.path.model_path,
            self.config.generator_model)
        self.discriminator_path = os.path.join(
            self.config.path.model_path,
            self.config.discriminator_model)
        self.classifier_path = os.path.join(
            self.config.path.model_path,
            self.config.classifier_model)
        if os.path.isfile(self.generator_path):
            self.generator = torch.load(self.generator_path)
        else:
            print("please first train the model")

        if os.path.isfile(self.classifier_path):
            self.classifier = torch.load(self.classifier_path)
        else:
            print("please first train the model")

        self.generator.cuda()
        self.classifier.cuda()

        self.data = VehiclesData(train=False)
        self.vehicle_state = VehicleState()

    def predict_position(self, vehicles: tuple, seq_len):
        """

        :param vehicles: vehicle info from get_current_data
        :param seq_len: predict sequence length
        :return:
        position_future: predicted position info [[[vehicle 1], [vehicle 2], ...]], vehicle 1: [lane number, position in road]   2*车辆的数量*预测的次数
        speed_future: predicted speed info [[vehicle 1, vehicle 2, ...]], vehicle 1: speed (float)
        vehicle_id: vehicle id [[vehicle 1, ...]]
        """
        vs, vid, t_0 = vehicles
        vehicles_in_intersection = {}
        initial_info = self.vehicle_state.get_some_vehicles_state(
            int(t_0) - 1, vid)
        now_state = np.array(initial_info)[:, 3:6]
        position_future = []
        vehicle_id = []
        speed_future = []

        self.classifier.eval()
        self.generator.eval()
        with torch.no_grad():
            for i in range(seq_len):
                vs = vs.cuda()
                _input = vs
                near_list = find_near_intersection_vehicle(
                    now_state, vid.tolist())

                real = self.vehicle_state.get_some_vehicles_state(
                    int(t_0) + i, vid.tolist())
                real = np.array(real)
                need_del = []

                for val in near_list:
                    need_del.append(np.argwhere(vid == val)[0])

                if len(need_del) != 0:
                    need_out_list = np.array(need_del)
                    vid = np.delete(vid, need_out_list)
                    _input = del_tensor(_input, need_out_list)
                    real = np.delete(real, need_out_list, axis=0)

                fake_o = self.generator(_input)
                fake_c = self.classifier(_input)

                fake_t = torch.zeros(
                    (fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
                fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o

                position = np.zeros((_input.shape[0], 3))

                position[:, 2] = real[:, 2]
                position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]

                speed = fake_o.detach().cpu().numpy()[:, -1]
                position_future.append(position[:, :2].tolist())
                speed_future.append(speed.tolist())
                vehicle_id.append(vid.tolist())

                vehicles_in_intersection, out_inter = update_intersection_vehicle(
                    vehicles_in_intersection)
                output = fake_t.detach().cpu().numpy()
                real_fake = real

                grip_now, vehicles_id = self.make_grid(
                    output, vid, real_fake)
                vs = grip_now
                vid = np.array(vehicles_id)
                now_state = np.zeros((grip_now.shape[0], 3))
                now_state[:, 1:] = np.array(grip_now)[:, :2]
                now_state[:, 0] = np.array(grip_now)[:, -2]

        return position_future, speed_future, vehicle_id

    def get_current_data(self, time: int, road_id=1) -> tuple:
        """get current data in road, which is tuple

        :param road_id: the road id of use, default is 1
        :param time: the index of test data
        :return: current vehicle data in one road: (vs, vid, t_0), (tensor, list, int)
        """
        assert isinstance(time, int), "time must be int"
        data = self.data[time]
        vs, vid, t_0 = data
        # x, y, road_id, lane_num, round(position_in_lane, 2), round(v_s, 2),
        vid = [v for v in vid]
        vid = np.array(vid)
        road_id = int(road_id)
        assert road_id not in vs[:, 2], "the road id is error"
        vehicle_id = vid[vs[:, -2] == road_id]
        state = vs[vs[:, -2] == road_id]
        res_state = torch.zeros((state.shape[0], 7))
        res_state[:, :-1], res_state[:, -1] = state[:, :-2], state[:, -1]

        return res_state, vehicle_id, t_0

    def make_grid(
            self,
            output: np.ndarray,
            vid: np.ndarray,
            vehicle_states: np.ndarray
            ):
        # output_type : [n_car, 3]  [lane_num, lane_position, speed]
        # vid : vehicle id list
        # vehicle_states : x, y, road_id, lane_num, position_lane, a, v_s, h_s
        assert output.shape[0] == vid.shape[0] and vid.shape[0] == vehicle_states.shape[0], "vehicle number need be " \
                                                                                            "consistent "
        # [position_lane, lateral_position, s, d, angle, lane_number, light, g_s, road_id, g_d]
        road_net = {}
        road_list = vehicle_states[:, 2]
        road_set = list(set(road_list))
        road_name_set = [
            self.config.selected_road[int(rid)] for rid in road_set]

        # 初始化字典
        for rname in road_name_set:
            road_net[rname] = {
                "vehicles": {},
                "road_id": rname
            }

        for key, _ in enumerate(vid):
            # 车辆id
            vehicle = vid[key]
            speed = output[key, 2]
            road_id = vehicle_states[key, 2]
            road_name = self.config.selected_road[int(road_id)]
            lane_num = output[key, 0]
            position_in_lane = output[key, 1]
            vinfo = {
                "id": vehicle,
                "info": {
                    'position': [0, 0],
                    "position_in_lane": position_in_lane,
                    "speed": speed,
                    "lane": "{}_{}".format(road_name, int(lane_num)),
                    "road": road_name,
                }
            }
            road_net[road_name]['vehicles'][vehicle] = vinfo

        grip_now, vehicles_id = get_data_from_map({"vehicles": road_net})
        grip_now = torch.tensor(grip_now, dtype=torch.float32)
        res_state = torch.zeros((grip_now.shape[0], 7))
        res_state[:, :-1], res_state[:, -1] = grip_now[:, :-2], grip_now[:, -1]
        return res_state, vehicles_id


if __name__ == '__main__':
    tp = TP()
    data = tp.get_current_data(10)
    print('data:',data)
    p, s, v = tp.predict_position(data, 10)



    print(p)
    print(s)
    print(v)
    for i in p[-1]:
        print(i[1])