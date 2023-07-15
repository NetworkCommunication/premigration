import numpy as np
import torch
import train_model
from data_process.vehicle_state import VehicleState
import json
import os

import math

import random



class System(object):

    def __init__(self):
        self.v_num=21
        self.r_num=11
        self.B=20
        self.N=174
        self.s=32
        self.s_t=3
        self.s_d=1/1000
        self.delta_t=0.1
        self.fi=1




        self.rsu = np.zeros(self.r_num)


        self.p = 0.2
        self.g = np.zeros((self.v_num, self.r_num))






    def set_rsu(self):
        num=self.r_num
        t=500/(num-1)
        t=int(t)
        r=0
        rsu=[]


        while r<=500 :
            rsu.append(r)
            r = r + t
        self.rsu=rsu

        return
    def get_vtr(self,vehicle):
        distance=[]
        connect=[]
        rsu=self.rsu
        for v in vehicle:
            j=0  #rsu
            for r in rsu:
                d=v-r
                if d<0:
                    d=d*(-1)
                if j==0:
                    dis=d
                    con=j
                if j>0 and d<dis:
                    dis = d
                    con = j
                j+=1

            distance.append(dis)
            connect.append(con)

        return distance,connect

    def get_usernum(self, connect):
        usernum= np.zeros(self.r_num, int)
        for con in connect:
            usernum[con]+=1
        return  usernum
    def get_c(self,usernum):
        c = []

        for unum in usernum:
            c_t = 24 - 4 * unum
            if c_t < 8:
                c_t = 8
            c.append(c_t)

        return c
    def tran_latency(self,vehicle , p):
        B=self.B
        N=self.N
        s=self.s
        s_t=self.s_t
        s_d=self.s_d
        delta_t=self.delta_t
        fi=self.fi

        rsu=self.rsu
        tran=np.zeros( (self.v_num,self.r_num) )

        distance, connect = sys.get_vtr(vehicle)
        usernum=self.get_usernum(connect)
        c=self.get_c(usernum)

        i=0
        for v in vehicle:

            j=0
            for r in rsu:

                tran_t= 1+   (  (self.p * self.g[i][j]) / (B*N)  )
                tran[i][j]=B * math.log(tran_t, 2)
                j+=1
            i+=1

        i = 0
        sum_l = 0
        sunm_real=0
        for v in vehicle:
            con = connect[i]
            min_l =0
            real_l=0

            j = 0
            x=0
            for r in rsu:
                d=r-rsu[con]
                if d<0:
                    d=(-1)*d

                ini_l=( s/tran[i][con] +s*d*s_d +s/c[j] )
                syn =fi* (1/delta_t)* ( s_t/tran[i][con] +s_t*d*s_d +s_t/c[j] )
                if j==0:
                    min_l=ini_l+syn
                    real_l=min_l-ini_l*p[i][j]
                if j>0 and (ini_l+syn)<min_l:
                    x=j
                    min_l=ini_l+syn
                    real_l=min_l-ini_l*p[i][j]

                j+=1


            sum_l += min_l
            sunm_real+=real_l

            i+=1

        print("Latency:",sum_l)
        print("Real latency:", sunm_real)

        ave_sum=sum_l/self.v_num
        ave_sum_real=sunm_real/ self.v_num

        return ave_sum,ave_sum_real
    def get_p(self,p_vehicle,c):

        B = self.B
        N = self.N
        s = self.s
        s_t = self.s_t
        s_d = self.s_d
        delta_t = self.delta_t
        fi = self.fi
        T = 10
        rsu = self.rsu
        p=np.zeros( (self.v_num, self.r_num) )
        tran = np.zeros((self.v_num, self.r_num) , int)
        eij=np.zeros( (self.v_num,self.r_num) )
        distance, connect = sys.get_vtr(p_vehicle)



        i = 0
        for v in p_vehicle:

            j = 0
            for r in rsu:
                tran_t = 1 + ((self.p * self.g[i][j]) / (B * N))
                tran[i][j] = B * math.log(tran_t, 2)
                j += 1
            i += 1


        i = 0
        for v in p_vehicle:
            con = connect[i]
            min_l =0


            j = 0
            for r in rsu:
                d=r-rsu[con]
                if d<0:
                    d=(-1)*d

                ini_l=(  s*d*s_d +s/c[j] )
                e=1/ini_l
                eij[i][j]=e
                j+=1


            i+=1

        max_num=  T /  (  s*s_d* 500/(self.r_num-1)  )
        max_num=int(max_num)



        for i in range(self.v_num):
            ei=eij[i]

            R=np.zeros(max_num)
            u=np.zeros(max_num,int)
            j=0
            for m in range(max_num):
                e_max=0
                u_max=0
                for j in range(self.r_num):

                    if   ei[j]>e_max  :
                        e_max=eij[i][j]
                        u_max = j
                ei[u_max] =-1
                R[m]=e_max
                u[m]=u_max

            e_mean=np.mean(R)
            p_num=   (max_num-1)* (  0.8*e_mean  - np.std(R)  ) /e_mean
            p_num=int(p_num) +1
            for p_n in range(p_num):

                u_n=u[p_n]
                p[i][u_n]=1


        return p





sys= System()












