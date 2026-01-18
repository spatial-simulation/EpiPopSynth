#!/usr/bin/python
# coding=utf-8
# @Author: ZhuKemin

"""
比较不同方法生成的合成人口的interdependency

"""

from collections import Counter
from operator import itemgetter

import pandas as pd
import numpy as np
import os



class DataAnalysisMotifDistribution:
    def __init__(self):
        self.path_pop = r"../Output/Populations"

    def __del__(self):
        pass

    def get_motif_distribution(self, mode):
        path_ = r"%s/%s"%(self.path_pop, mode)
        filelist = list(os.walk(path_))[0][2]
        filelist = [os.path.join(path_,fn) for fn in filelist]

        res = []
        for i,fn in enumerate(filelist):
            print (i)
            mat = pd.read_csv(fn).to_numpy().astype(int)
            hid,hindex = np.unique(mat[:,1], return_index=True)
            mat_splited = np.split(mat, hindex[1:])                                                           # group individual by household_id

            household_list = []
            for household in mat_splited:                                                                     # analysis results of multiple repeat experiment
                household_structure = Counter(map(tuple, household[:, (2,3)].tolist()))                       # get combination of (gender,age)
                household_structure = sorted(household_structure.items(), key=itemgetter(0))
                household_list.append(tuple(household_structure))
            household_count = Counter(household_list)                                                         # dict that recoed frequency of households
            hcodes,hcounts = np.array(sorted(household_count.items(), key=itemgetter(1), reverse=True)).T     # count household structure
            res.append([hcodes,hcounts])
        return res

    def run(self):
        for mode in ["mh","di","ipf"]:
            res = self.get_motif_distribution(mode)
            np.save("../Output/Results/%s.npy"%(mode), res)



if __name__ == '__main__':
    DAD = DataAnalysisMotifDistribution()
    DAD.run()