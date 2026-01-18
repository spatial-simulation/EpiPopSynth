#!/usr/bin/python
# coding=utf-8

# from mpi4py import MPI
import sys

# comm = MPI.COMM_WORLD
# comm_rank = comm.Get_rank()
# comm_size = comm.Get_size()
# print("comm_rank:",comm_rank)
# print("comm_size:",comm_size)

import matplotlib.pyplot as plt
# from scipy.optimize import least_squares
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import scipy
import time

# ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust-constr", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"]


config = {
    "font.family": "sans-serif",
    "font.serif": ["Helvetica"],
    # "font.serif": ["Times New Roman"],
    # "font.serif": ["SimHei"],
    "font.size": 12,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
}
plt.rcParams.update(config)


class GridHouseholdSimulation():
    def __init__(self):
        self.fn_household_motif = r"./Data/mat_household_motif.xls"
        self.fn_demographic_community = r"./Data/demographic_community.xls"
        self.fn_household_size_count = r"./Data/household_size_count.csv"
        self.fn_gender = r"./Data/result_szxjy_sq_homegender.csv"

    def __del__(self):
        pass

    def fun_bak(self, weight_age, weight_gender, weight_hsize, y_age, y_gender, y_hsize):
        fitness = lambda x: np.sqrt((np.r_[x@weight_age    - y_age,
                                           x@weight_gender - y_gender,
                                           x@weight_hsize  - y_hsize,
                                           ]**2).mean())
        return fitness

    def fun(self, X):
        weight_age, weight_gender, weight_hsize = self.W
        y_age, y_gender, y_hsize = self.Y

        fitness = sum(np.r_[X@weight_age    - y_age,
                            X@weight_gender - y_gender,
                            X@weight_hsize  - y_hsize,
                            ]**2)
        self.fit = fitness
        return fitness

    def optimize_motif(self, zone_id, x_init, epsilon=0.5):
        lb = np.zeros(x_init.shape)
        ub = (x_init*(1+epsilon)).astype(int)
        np.putmask(ub, ub<10, 10)
        bnds = np.c_[lb,ub]

        # res = minimize(self.fun, x0=x_init, method='SLSQP', bounds=bnds, callback=self.callbackF, options={"maxiter":5})
        res = minimize(self.fun, x0=x_init, method='SLSQP', bounds=bnds, callback=self.callbackF)
        # res = minimize(self.fun, x0=x_init, method='BFGS', bounds=bnds, callback=self.callbackF, options={"maxiter":200})
        # res = minimize(self.fun, x0=x_init, method='trust-exact', bounds=bnds, options={"maxiter":5})
        return res

    def optimize_motif_bak(self, zone_id, x_init, weight_age, weight_gender, weight_hsize, y_age, y_gender, y_hsize):
        n_min_ptype = min(np.r_[y_age,y_gender])
        bnds = np.c_[np.zeros(x_init.shape), np.r_[x_init[:-1]*50, n_min_ptype]]
        res = minimize(self.fun(weight_age, weight_gender, weight_hsize, y_age, y_gender, y_hsize), x0=x_init, method='SLSQP', bounds=bnds)
        best_x = np.round(res.x).astype(int)
        return best_x

    def read_gender(self, fn):
        gender_count = pd.read_csv(fn, delimiter=",")

        dict_gender_rate = {}
        for zone,gp in gender_count.groupby("zone"):
            zone = int(zone[:17])   # "44030300100020000--桂园"  →  "44030300100020000"
            n_female = sum(gp[gp["gender_group"]=="女"]["home_num_total"])
            n_male   = sum(gp[gp["gender_group"]=="男"]["home_num_total"])

            if n_female+n_male:
                dict_gender_rate[zone] = np.array([n_female, n_male])/(n_female+n_male)
            else:
                dict_gender_rate[zone] = np.array([0.46582231, 0.53417769])  # overall gender ratio
        return dict_gender_rate

    def read_household_size(self, fn):
        hsize_count = pd.read_csv(fn, delimiter=",").to_numpy()
        n_pop = hsize_count[:,0]@hsize_count[:,1]
        hsize_rate = hsize_count[:,1]/n_pop
        return hsize_rate

    def callbackF(self, X):
        # weight_age, weight_gender, weight_hsize = self.W
        # y_age, y_gender, y_hsize = self.Y

        # fitness = sum(np.r_[X@weight_age    - y_age,
        #                     X@weight_gender - y_gender,
        #                     X@weight_hsize  - y_hsize,
        #                     ]**2)
        # self.logf.append(fitness)
        self.logf.append(self.fit)

    def run(self):
        df_motif_count = pd.read_excel(self.fn_household_motif)
        x_init_rate    = df_motif_count.iloc[:,  1   ].values
        weight_gender  = df_motif_count.iloc[:,  2: 4].values
        weight_age     = df_motif_count.iloc[:,  4:17].values
        weight_hsize   = df_motif_count.iloc[:, 17:  ].values

        demographics     = pd.read_excel(self.fn_demographic_community)
        hsize_rate       = self.read_household_size(self.fn_household_size_count)
        dict_gender_rate = self.read_gender(self.fn_gender)

        comm_rank=1
        if comm_rank<=len(demographics):
            demographics_ = demographics.iloc[comm_rank]
            zone_id  = demographics_["SQCODE"]
            n_zone   = demographics_["总数"]

            t1 = time.time()
            if n_zone==0:
                print ("No data in zone: %s !"%(zone_id))
            else:
                y_age    = demographics_[["0-2岁", "3-5岁", "6-11岁", "12-14岁", "15-17岁", "18-24岁", "25-29岁", "30-39岁", "40-49岁", "50-59岁", "60-69岁", "70-79岁", "80岁+"]].to_numpy().astype(int)
                y_gender = np.round(dict_gender_rate[zone_id]*n_zone).astype(int)
                y_hsize  = np.round(hsize_rate*n_zone).astype(int)
                x_init   = (x_init_rate*n_zone).astype(int)

                self.logf = []
                self.W = [weight_age, weight_gender, weight_hsize]
                self.Y = [y_age, y_gender, y_hsize]

                res = self.optimize_motif(zone_id, x_init)
                best_x = np.round(res.x).astype(int)
                # print (time.time()-t1)

                print ("------")

                plt.plot(self.logf, color="royalblue", lw=1.5)

                self.logf = np.array(self.logf)
                plt.fill_between(self.logf*0.95, self.logf*1.05, color="cornflowerblue", alpha=0.2)

                plt.xlim(0, len(self.logf))
                plt.xlabel("Iteration")
                plt.ylabel("Objective")

                plt.grid(color="lightgray", ls="-.", alpha=0.8)
                # plt.show()
                plt.savefig("./Output/optimization.svg", format="svg")

                # np.save(r"./Output/BestX Community/%s.npy"%(zone_id), best_x)
                # self.plot_age_comparison(best_x, weight_age, y_age)


    def plot_age_comparison(self, best_x, weight_age, y_age):
        plt.close()
        age_obs = best_x@weight_age
        age_sim = y_age

        width = 0.35
        n = np.arange(len(age_obs))
        fig, ax = plt.subplots()
        rects1 = ax.bar(n-width/2, age_sim, width, color="#708090", label='Simulation' ,  edgecolor='lightgray', zorder=10)
        rects2 = ax.bar(n+width/2, age_obs, width, color="#FFA07A", label='Observation',  edgecolor='lightgray', zorder=10)
        ax.set_xticks(n)
        # ax.set_xticklabels(labels)
        ax.grid(linestyle='-.', zorder=0)
        # plt.show()
        # plt.savefig("./Output/margin_age.svg", format="svg")

    def plot_hsize_comparison(self, best_x, typical_motif, hsize_count):
        plt.close()
        typical_motif["x"] = best_x
        hsize_obs = hsize_count["count"]
        hsize_sim = typical_motif.groupby(["hsize"]).agg({"x":"sum"})["x"].values
        labels = self.hsize_label_dict.values()

        width = 0.35
        n = np.arange(len(labels))
        fig, ax = plt.subplots()
        rects1 = ax.bar(n-width/2, hsize_sim, width, color="#708090", label='Simulation' ,  edgecolor='lightgray', zorder=10)
        rects2 = ax.bar(n+width/2, hsize_obs, width, color="#FFA07A", label='Observation',  edgecolor='lightgray', zorder=10)
        ax.set_xticks(n)
        ax.set_xticklabels(labels)
        ax.grid(linestyle='-.', zorder=0)
        plt.show()






if __name__ == '__main__':
    GHS = GridHouseholdSimulation()
    GHS.run()


