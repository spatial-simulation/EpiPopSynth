#!/usr/bin/python
# coding=utf-8
# @Author: ZhuKemin

"""
对不同来源的Survey数据进行预处理, 仅保留其 个体编号、家庭编号、性别、年龄
并对年龄进行重新分组
"""

import pandas as pd
import numpy as np


class DataPreprocessSurvey:
    def __init__(self, fn_survey, age_bounds, max_hsize, col_p="person_id", col_h="household_id", col_g="gender", col_a="age"):
        self.age_bounds = age_bounds
        self.max_hsize = max_hsize
        self.df_survey = self._read_survey(fn_survey, col_p, col_h, col_g, col_a)

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

    def run(self):
        df_survey_structured = self.process_survey(self.df_survey)
        df_margin = self.get_margin(df_survey_structured)
        df_survey_structured.to_csv("../Property/Processed Survey Data/survey_family_only_structured.csv", index=False)
        df_margin.to_csv("../Property/Margin/margin_family_only_survey.csv", index=False)


if __name__ == '__main__':
    fn_survey = r"../Data/Raw Survey Data/Survey Shenzhen/survey_family_only.csv"
    age_bounds = np.array([10,20,30,40,50,60,70,80])
    max_hsize = 6

    DPS = DataPreprocessSurvey(fn_survey, age_bounds, max_hsize)
    DPS.run()