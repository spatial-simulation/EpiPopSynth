#!/usr/bin/python
# coding=utf-8
# @Author: ZhuKemin

"""
对预处理过的survey数据以家庭为单位进行抽样并存储
"""

import pandas as pd
import numpy as np


class DataPreprocessSampler:
    def __init__(self, fn_survey):
        self.df_survey = pd.read_csv(fn_survey)

    def __del__(self):
        pass

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
        df_survey["age"] = np.searchsorted(self.age_bounds, df_survey["age"], side="right")
        return df_survey

    def get_margin(self, df_survey_structured):
        dim_a = len(self.age_bounds)+1
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

    def run(self, n_samp=1, samp_rate=0.01):
        n_h_select = int(round(self.df_survey["hid"].nunique()*samp_rate))

        for i in range(n_samp):
            print (i)
            h_select = np.random.choice(self.df_survey["hid"].unique(), n_h_select)
            df_sampled = self.df_survey.query("hid in @h_select")
            df_sampled.to_csv(r"../Property/Sampled Survey Data/samp_%03i.csv"%(i))


if __name__ == '__main__':
    fn_survey = r"../Property/Processed Survey Data/survey_family_only_structured.csv"
    DPS = DataPreprocessSampler(fn_survey)
    DPS.run(n_samp=100)