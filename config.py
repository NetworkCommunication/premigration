#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : config.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/19 9:12

import os

__all__ = ['Config']


class Path(object):
    """Path for file

    Attributes:
        __aba_path (String): the absolute path of this file
        __root_dir (String): the absolute basic path of this project
        sumo_path (String): the absolute path of sumo file
        data_path (String): the absolute path of data file
    """

    def __init__(self):
        self.__aba_path = os.path.abspath(__file__)
        self.__root_dir = os.path.dirname(self.__aba_path)
        self.sumo_path = os.path.join(self.__root_dir, 'assets\\sumo')
        self.data_path = os.path.join(self.__root_dir, 'assets\\data')
        self.model_path = os.path.join(self.__root_dir, 'assets\\model')

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)


# ---- basic configure in our project ----
class Config(object):
    """This class integrates some basic setting for our project

    Attributes:
        path (Path): the path object.
        traffic_filename (string): Road traffic filename.
        trafficLight_filename (string): Traffic light filename.
        vehicle_filename (string): Vehicle information filename.
        roadMap_filename (string): Road map filename.
    """

    def __init__(self):
        self.path = Path()
        self.traffic_filename = 'road_traffics.json'
        self.trafficLight_filename = 'traffic_lights.json'
        self.vehicle_filename = 'vehicles.json'
        self.roadMap_filename = 'road_map.json'
        self.traffic_flow = 'traffic_flow.json'
        self.vehicle_info = 'vehicle_info.json'
        self.vehicle_state = 'vehicle_state.json'
        self.classifier_model = 'classifier.pth'
        self.generator_model = 'generator.pth'
        self.discriminator_model = 'discriminator.pth'
        self.min_length = 5  # minimal length of vehicle in sumo
        self.min_gap = 2.5  # minimal gap of vehicle in sumo
        self.visible_area = 50  # the visible area of vehicle equipped with OBU
        self.history_window = 5  # length of history window
        self.selected_road = ['gneE3', '-gneE2', '-gneE3', 'gneE2', '-gneE5', 'gneE5', '-gneE6', 'gneE6']
        self.output_window = 5
        self.road_index = 1  # road id
