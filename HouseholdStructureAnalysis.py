#!/usr/bin/python
# coding=utf-8
# @Author: ZhuKemin


"""
从2016人口调查数据中获取各类典型家庭结构的 数量、 属性矩阵

"""

from collections import Counter
from operator import itemgetter

import numpy.lib.recfunctions as rfn
import pandas as pd
import numpy as np
import pickle

from warnings import filterwarnings
filterwarnings('ignore', category=np.VisibleDeprecationWarning)
filterwarnings('ignore', category=FutureWarning)
filterwarnings('ignore', category=UserWarning)


class HouseholdStructureAnalysis():
    def __init__(self):
        self.age_bounds = np.array([10,20,30,40,50,60,70,80])
        self.max_hsize = 6

        self.fn_survey = r"./Data/survey_family_only.csv"
        self.fn_output = r"./Property/mat_motif_family_only.xls"

        # self.fn_survey = r"./Data/survey_collective_only.csv"
        # self.fn_output = r"./Property/mat_motif_collective_only.xls"
        self.fn_output_margin = r"./Property/margin_survey.xls"
        self.fn_output_pcount = r"./Property/ptype_count.pkl"

        assert self.age_bounds[0]>0

    def __del__(self):
        pass

    def count_n_cross_level(self, household):
        """ 计算每种家庭结构的规模/性别人数/年龄人数 """
        arr_age = np.zeros(len(self.age_bounds)+1, dtype=int)
        arr_gender = np.zeros(2, dtype=int)
        arr_hsize  = np.zeros(self.max_hsize, dtype=int)

        hsize = 0
        for (p_gender,p_age),pcount in household:
            arr_gender[p_gender] += pcount
            arr_age[p_age] += pcount
            hsize += pcount
        hsize = min(hsize, self.max_hsize)-1
        arr_hsize[hsize] = 1
        # arr_hsize[min(hsize, self.max_hsize)-1] = 1
        arr_n_cross_level = np.r_[hsize,arr_gender,arr_age,arr_hsize]
        return arr_n_cross_level

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
        ages = mat_survey[:,2]
        age_groups = np.searchsorted(self.age_bounds, ages, side="right")
        # mat_survey[:,2] = age_groups                                                              # replace age with age-group
        mat_survey = np.c_[mat_survey, age_groups]                                                  # append mat_survey with age-group

        mat_survey = mat_survey[mat_survey[:,0].argsort()]                                          # sort by household_id
        household_id, household_index = np.unique(mat_survey[:,0],return_index=True)
        mat_survey_splited = np.split(mat_survey, household_index[1:])                              # group individual by household_id

        household_list = []
        for household in mat_survey_splited:
            household_structure = Counter(map(tuple,household[:,(1,2)].tolist()))                   # get combination of (gender,age)
            household_structure = sorted(household_structure.items(), key=itemgetter(0))
            household_list.append(tuple(household_structure))
        household_count = Counter(household_list)                                                   # dict that recoed frequency of households
        mat_household = self.get_structured_array(household_count)                                  # organized as an structured_array (htype-count)
        return mat_household

    def get_n_motif_prop(self, mat_household, prop_threahold=0.99):
        # 选取cdf覆盖prop_threahold比例的motif数量
        household_prop = mat_household["h_count"]/sum(mat_household["h_count"])
        household_cums = np.cumsum(household_prop)
        n_motif_prop = np.searchsorted(household_cums, prop_threahold)
        return n_motif_prop

    def get_n_motif_nonzero_ptype(self, mat_household):
        # 提取survey数据中所有的ptype类型
        ptype_concat = np.concatenate(list(map(eval, mat_household["h_code"])))[:,0]
        ptype_unique = np.unique(ptype_concat)

        # 确保每种个体类型都有的none-zero-cell的motif
        n_motif_nonzero_ptype = max([list(ptype_concat).index(pt) for pt in ptype_unique])
        n_motif_nonzero_ptype = np.searchsorted(np.cumsum(mat_household["h_size"]), n_motif_nonzero_ptype, side="right")
        return n_motif_nonzero_ptype

    def get_n_motif_nonzero_hsize(self, mat_household):
        # 确保每种户规模类型都有的none-zero-cell的motif
        n_motif_nonzero_hsize = max([list(mat_household["h_size"]).index(hsz) for hsz in np.unique(mat_household["h_size"])])
        return n_motif_nonzero_hsize

    def get_motif_label(self, mat_motif):
        age_bounds_ = np.r_[[0], self.age_bounds, ["%i+"%(self.age_bounds[-1])]]

        h_labels = []
        for h_code in map(eval, mat_motif["h_code"]):
            plist = []
            for [gender,age],n in h_code:
                gender = "male" if gender else "female"
                age = "%s-%s"%(age_bounds_[age],age_bounds_[age+1])
                plist += ["%s(%s)"%(gender,age)]*n
            h_labels.append(" ".join(plist))
        return h_labels

    def select_motif(self, mat_household):
        """ 确定满足覆盖率需求且能避免 zero-cell problem 的最小 motifs 数量 """
        n_motif_prop = self.get_n_motif_prop(mat_household)
        n_motif_nonzero_ptype = self.get_n_motif_nonzero_ptype(mat_household)
        n_motif_nonzero_hsize = self.get_n_motif_nonzero_hsize(mat_household)
        n_motif = max(n_motif_prop, n_motif_nonzero_ptype, n_motif_nonzero_hsize)
        mat_motif = mat_household[:n_motif+1]

        mat_motif["x_init" ] = mat_motif["h_count"]/sum(mat_motif["h_count"])
        mat_motif["h_label"] = self.get_motif_label(mat_motif)
        return mat_motif

    def get_demographic(self, mat_survey, mat_household):
        margin_age = np.bincount(mat_survey[:,3])
        margin_gender = np.bincount(mat_survey[:,1])
        margin_hsize = np.bincount(np.repeat(mat_household["h_size"], mat_household["h_count"]))
        margin = np.r_[margin_gender,margin_age,margin_hsize]

        dim_a = len(self.age_bounds)+1
        dim_s = self.max_hsize
        colnames_a = ["n_a%i"%(a) for a in range(dim_a)]
        colnames_s = ["n_s%i"%(a) for a in range(dim_s)]
        colnames_g = ["n_female","n_male"]
        colnames = np.r_[colnames_g, colnames_a, colnames_s]
        mat_margin_survey = pd.DataFrame(margin.reshape(1,-1), columns=colnames)
        return mat_margin_survey

    def count_person_type(self, mat_survey):
        ages = mat_survey[:,2]
        age_groups = np.searchsorted(self.age_bounds, ages, side="right")
        mat_survey = np.c_[mat_survey, age_groups]
        count_ptype = Counter(map(tuple,mat_survey[:,(1,3)].tolist()))
        return count_ptype

    def run(self):
        """total 6557738 individual"""
        mat_survey = pd.read_csv(self.fn_survey)[["household_id","gender","age"]].to_numpy()
        dict_pcount = self.count_person_type(mat_survey)
        with open(self.fn_output_pcount, "wb") as fh:
            pickle.dump(dict_pcount, fh)

        # mat_household = self.count_household_structure(mat_survey)
        # mat_motif = self.select_motif(mat_household)
        # pd.DataFrame(mat_motif).to_excel(self.fn_output, index=False)

        # mat_margin_survey = self.get_demographic(mat_survey, mat_household)
        # mat_margin_survey.to_excel(self.fn_output_margin, index=False)


if __name__ == '__main__':
    HSA = HouseholdStructureAnalysis()
    HSA.run()
