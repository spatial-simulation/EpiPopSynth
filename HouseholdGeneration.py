#!/usr/bin/python
# coding=utf-8
# @Author: ZhuKemin

# 注意 mat_motif 中的 hsize 起始为0


"""
根据 motif 和 best_x 生成 pop
"""

import pandas as pd
import numpy as np
import pickle
import os


class HouseholdGeneration:
    def __init__(self):
        self.fn_household_motif = r"./Property/mat_motif_family_only.xls"
        self.fn_best_x = r"./Property/Res-v2.pkl"
        self.path_output = r"./Output/Populations"
        self.n_agent = 17376188

    def _decode_motif(self, str_motif):
        gender_age = np.concatenate(list(map(lambda x: np.repeat(x[0],x[1]), eval(str_motif))))
        gender_age = gender_age.reshape(-1,2)
        return gender_age

    def get_pop_mh(self, p_arr):
        hsizes = np.fromiter(map(len,p_arr), dtype=int)
        hids = np.repeat(np.arange(len(hsizes)), hsizes)
        pids = np.arange(len(hids))
        mat_person = np.c_[pids, hids, np.concatenate(p_arr)]
        return mat_person

    def run(self, n_repeat=100):
        df_motif_count = pd.read_excel(self.fn_household_motif)
        motifs = df_motif_count.h_code.values

        with open(self.fn_best_x, "rb") as fh:
            best_x = pickle.load(fh)["x"]
        
        p = best_x/sum(best_x)                                     # 选取概率
        hsize_avg = sum((df_motif_count.h_size.values+1)*p)        # 平均户规模      <-------
        N = (self.n_agent/hsize_avg).astype(int)                   # 选取次数

        # 初始化输出路径
        if not os.path.exists(r"./%s/%s"%(self.path_output, "mh")):
            os.makedirs(r"./%s/%s"%(self.path_output, "mh"))

        # 开展重复实验并存储实验结果
        for i in range(n_repeat):
            fn_output_mh  = os.path.join(self.path_output, "mh/pop_%03i.csv"%(i))

            # 选取 Household motifs
            motifs_ = np.random.choice(motifs, size=N, p=p)
            p_arr = list(map(self._decode_motif, motifs_))
            mat_person_mh  = self.get_pop_mh (p_arr)
            pd.DataFrame(mat_person_mh,  columns=["pids","hids","gender","age"]).to_csv(fn_output_mh , index=False)

if __name__=='__main__':
    HG = HouseholdGeneration()
    HG.run()
