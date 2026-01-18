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


class DataAnalysis:
    def __init__(self, fn_survey, age_bound, max_hsize, col_p="person_id", col_h="household_id", col_g="gender", col_a="age"):
        self.age_bound = age_bound
        self.max_hsize = max_hsize
        self.df_survey = self._read_survey(fn_survey, col_p, col_h, col_g, col_a)

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

        dim_a = len(self.age_bound)+1
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


    # def run(self):
    #     fn = r"../Output/Populations/mh/pop_000.csv"
    #     self.get_interdepend_synthetic(fn)

        





    def _read_survey(self, fn_survey, col_p, col_h, col_g, col_a):
        suffix = fn_survey.split(".")[-1]
        columns = [col_p,col_h,col_g,col_a]
        if suffix=="csv":
            df_survey = pd.read_csv(fn_survey, usecols=columns)
        elif suffix=="xls":
            df_survey = pd.read_excel(fn_survey, usecols=columns)
        df_survey = df_survey.rename(columns={col_p:"pid", col_h:"hid", col_g:"gender", col_a:"age"})
        df_survey = df_survey[["pid","hid","gender","age"]]
        return df_survey

    def process_survey(self, df_survey):
        """ survey数据的排序和年龄重分组 """
        df_survey = df_survey.sort_values("pid")
        df_survey = df_survey.reset_index(drop=True)
        df_survey["age"] = np.searchsorted(self.age_bound, df_survey["age"], side="right")
        return df_survey

    def get_margin(self, df_survey_structured):
        dim_a = len(self.age_bound)+1
        dim_s = self.max_hsize
        colnames_a = ["n_a%i"%(a) for a in range(dim_a)]
        colnames_s = ["n_s%i"%(a) for a in range(dim_s)]
        colnames_g = ["n_female","n_male"]

        hsizes = df_survey_structured.groupby("hid").apply(len).values
        np.putmask(hsizes, hsizes>self.max_hsize, self.max_hsize)
        margin_hsize = np.bincount(hsizes)[1:]

        margin_age = np.bincount(df_survey_structured.age)
        margin_gender = np.bincount(df_survey_structured.gender)
        margin = np.r_[margin_gender,margin_age,margin_hsize]
        df_margin = pd.DataFrame([margin], columns=np.r_[colnames_g,colnames_a,colnames_s])
        return df_margin

    def run(self):
        df_survey_structured = self.process_survey(self.df_survey)
        # print (max(np.bincount(df_survey_structured.hid.values)))
        print (df_survey_structured)
        exit()



        # df_margin = self.get_margin(df_survey_structured)
        # df_survey_structured.to_csv("../Property/Processed Survey Data/survey_family_only_structured.csv", index=False)
        # df_margin.to_csv("../Property/Margin/margin_family_only_survey.csv", index=False)



if __name__ == '__main__':
    # fn_survey = r"../Data/Raw Survey Data/Survey Shenzhen/survey_family_only.csv"
    fn_survey = r"../Data/Raw Survey Data/Survey Shenzhen/survey.csv"
    age_bound = np.array([10,20,30,40,50,60,70,80])
    max_hsize = 6

    DA = DataAnalysis(fn_survey, age_bound, max_hsize)
    DA.run()