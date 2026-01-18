#!/usr/bin/python
# coding=utf-8
# @Author: ZhuKemin


"""
人口合成 PopulationSynthsis.py 的前步骤
用来优化迭代地计算各类 Household Motif 的权重, 以fit人口边缘分布
设计为单个空间单元 (或非空间) 的优化 
处理空间显式的人工人口 需要在高性能环境中并行 (见上一版本 SubdistrictHouseholdSimulation.py)

与原始 survey 数据的差异作为罚函数计入目标函数
"""

from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
from collections import defaultdict

import pandas as pd
import numpy as np
import pickle
import os

from scipy.stats import entropy


class HouseholdOptimization:
    def __init__(self, city_name):
        self.city_name = city_name

        self.fn_household_motif = r"./Property/mat_motif_family_only.xls"
        self.fn_margin_survey = r"./Property/margin_survey.xls"

        self.age_bounds = np.array([10,20,30,40,50,60,70,80])
        self.max_hsize = 6

        path_output = r"./Property/"
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        self.fn_output = os.path.join(path_output, "Res-v3.pkl")

    def __del__(self):
        pass

    def cal_JS(self, p_, q_):
        p = p_/np.sum(p_)
        q = q_/np.sum(q_)
        m = 0.5*(p+q)
        jsd = 0.5*(entropy(p, m)+entropy(q, m))
        return jsd

    def fun_(self, x, w_a, w_g, w_s, y_a, y_g, y_s, phi=1):
        loss_a = (y_a-x@w_a)
        loss_g = (y_g-x@w_g)
        loss_s = (y_s-x@w_s)
        loss_m = (x-self.x_init_survey)*phi
        obj = np.r_[loss_a, loss_g, loss_s, loss_m]

        mse_a = np.sum((loss_a/sum(y_a))**2)
        mse_g = np.sum((loss_g/sum(y_g))**2)
        mse_s = np.sum((loss_s/sum(y_s))**2)
        mse_m = np.sum((loss_m/sum(self.x_init_survey))**2)

        self.dict_opt_progress["age"   ].append(mse_a)
        self.dict_opt_progress["gender"].append(mse_g)
        self.dict_opt_progress["hsize" ].append(mse_s)
        self.dict_opt_progress["motif" ].append(mse_m)
        self.dict_opt_progress["cost"  ].append(mse_a+mse_g+mse_s+mse_m)
        return obj

    def fun(self, w_a, w_g, w_s, y_a, y_g, y_s, x):
        fitness = lambda x: self.fun_(x, w_a, w_g, w_s, y_a, y_g, y_s)
        return fitness

    def con(self):
        cons = {'type':'ineq', 'fun':lambda x: min(x)}
        return cons

    def optimize_motif(self, x_init, w_a, w_g, w_s, y_a, y_g, y_s):
        bnds = np.c_[np.zeros(x_init.shape), x_init*10]
        cons = self.con()
        res = least_squares(self.fun(w_a, w_g, w_s, y_a, y_g, y_s, x_init), x0=x_init, bounds=[0, max(x_init)*50], verbose=2, ftol=1e-16, xtol=1e-16, gtol=1e-14, max_nfev=50)
        return res

    # @profile
    def run(self):
        # 读取原始的x比例和weight
        mat_motif = pd.read_excel(self.fn_household_motif, sheet_name=0)
        dim_a = len(self.age_bounds)+1
        dim_s = self.max_hsize
        colnames_a = ["n_a%i"%(a) for a in range(dim_a)]
        colnames_s = ["n_s%i"%(a) for a in range(dim_s)]
        colnames_g = ["n_female","n_male"]
        w_g = mat_motif[colnames_g].values
        w_a = mat_motif[colnames_a].values
        w_s = mat_motif[colnames_s].values

        # 读取要拟合的marginal
        mat_margin_survey = pd.read_excel(self.fn_margin_survey)
        colnames_a = ["n_a%i"%(a) for a in range(dim_a)]
        colnames_s = ["n_s%i"%(a) for a in range(dim_s)]
        y_a = mat_margin_survey[colnames_a].values[0]
        y_s = mat_margin_survey[colnames_s].values[0]
        y_g = mat_margin_survey[colnames_g].values[0]
        self.n_agent = np.sum(y_g)

        # 比例与数量转换
        w_np = mat_motif.h_size.values+1   # hszie start from 1
        r_init = mat_motif["x_init"].values
        x_init = np.round(r_init*(self.n_agent/sum(r_init*w_np))).astype(int)
        self.x_init_survey = x_init
        x_init = x_init*np.random.lognormal(0, 0.5, x_init.shape)

        # 初始化记录文件
        self.dict_opt_progress = defaultdict(list)
        res = self.optimize_motif(x_init, w_a, w_g, w_s, y_a, y_g, y_s)
        best_x = np.round(res.x).astype(int)

        res = dict(res)
        res["progress"] = self.dict_opt_progress
        with open(self.fn_output, "wb") as fh:
            pickle.dump(dict(res), fh)




if __name__ == '__main__':
    for city_name in ["Shenzhen"]:
        HO = HouseholdOptimization(city_name)
        HO.run()


