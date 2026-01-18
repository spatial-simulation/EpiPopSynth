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


class DataAnalysisInterdependency:
    def __init__(self):
        pass

    def __del__(self):
        pass

    def get_structured_array(self, household_count):
        """ 用来生成结构化数组 """
        hcodes,hcounts = np.array(sorted(household_count.items(), key=itemgetter(1), reverse=True)).T
        mat_cross_level = np.array([self.count_n_cross_level(household) for household in hcodes])
        x_init = np.zeros(hcodes.shape[0], dtype=float)
        hcodes = np.fromiter(map(str, hcodes), dtype="U255")
        hcounts = hcounts.astype(int)
        hlabels = np.zeros(hcodes.shape[0], dtype="U255")

        dim_a = len(self.age_bounds)+1
        dim_s = self.max_hsize
        colnames_a = ["n_a%i"%(a) for a in range(dim_a)]
        colnames_s = ["n_s%i"%(a) for a in range(dim_s)]
        colnames = np.r_[["h_size","n_female","n_male"], colnames_a, colnames_s]

        h_dtype = np.r_[[("h_code", "U255"), ("x_init", "float"), ("h_count", "int32")], [(cname, "int16") for cname in colnames], [("h_label", "U255")]]
        h_dtype = np.dtype([tuple(_) for _ in h_dtype])

        mat_household = np.c_[hcodes.reshape(-1,1), x_init.reshape(-1,1), hcounts.reshape(-1,1), mat_cross_level, hlabels.reshape(-1,1)]
        mat_household = rfn.unstructured_to_structured(mat_household, h_dtype)
        return mat_household

    def count_household_structure(self, mat_survey):
        mat_survey = mat_survey[mat_survey[:,0].argsort()]                                                # sort by household_id
        household_id, household_index = np.unique(mat_survey[:,0],return_index=True)
        mat_survey_splited = np.split(mat_survey, household_index[1:])                                    # group individual by household_id

        household_list = []
        for household in mat_survey_splited:
            household_structure = Counter(map(tuple,household[:,(1,2)].tolist()))                         # get combination of (gender,age)
            household_structure = sorted(household_structure.items(), key=itemgetter(0))
            household_list.append(tuple(household_structure))
        household_count = Counter(household_list)                                                         # dict that recoed frequency of households
        mat_household = self.get_structured_array(household_count)                                        # organized as an structured_array (htype-count)
        return mat_household

    def get_interdepend_synthetic(self, fn):
        mat = pd.read_csv(fn).to_numpy().astype(int)
        hid,hindex = np.unique(mat[:,1], return_index=True)
        mat_splited = np.split(mat, hindex[1:])                                                           # group individual by household_id

        household_list = []
        for household in mat_splited:
            household_structure = Counter(map(tuple, household[:, (2,3)].tolist()))                       # get combination of (gender,age)
            household_structure = sorted(household_structure.items(), key=itemgetter(0))
            household_list.append(tuple(household_structure))
        household_count = Counter(household_list)                                                         # dict that recoed frequency of households
        hcodes,hcounts = np.array(sorted(household_count.items(), key=itemgetter(1), reverse=True)).T     # count household structure

    def run(self):
        fn = r"../Output/Populations/mh/pop_000.csv"
        self.get_interdepend_synthetic(fn)




if __name__ == '__main__':
    DAD = DataAnalysisInterdependency()
    DAD.run()