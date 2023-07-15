#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File : tools.py
# @Project : TP
# @Author : RenJianK
# @Time : 2022/5/19 10:00

def get_lane_number(lane_name: str):
    lane_number = lane_name.split('_')[-1]
    return int(lane_number)
