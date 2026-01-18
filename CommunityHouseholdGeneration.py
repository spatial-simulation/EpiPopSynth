#!/usr/bin/python
# coding=utf-8


import pandas as pd
import numpy as np
import os

SAMP  = False
SRATE = 0.1

class SubdistrictHouseholdGeneration():
    def __init__(self):
        self.fn_household_motif = r"./Property/mat_motif_family_only.xls"
        self.fn_best_x = r"./Res-v2.pkl"
        self.fn_output = r"./Output/mat_person_household.csv"

    def __del__(self):
        pass

    def run(self):
        df_motif_count = pd.read_excel(self.fn_household_motif)
        motifs = np.array([[[p_gender,p_age,pcount] for (p_gender,p_age),pcount in eval(motif_str)] for motif_str in df_motif_count["h_code"].values])
        hsize  = df_motif_count[["n_female","n_male"]].apply(sum,axis=1).values

        hid_start = 0
        pid_start = 0
        mat_person = []
        for root,dirs,files in os.walk(self.data_path):
            for zone_fn in files:
                zone_id = int(zone_fn.split('.')[0])
                print (zone_id)

                X = np.load(os.path.join(self.data_path, zone_fn))
                X = np.round(X*SRATE).astype(int) if SAMP else X

                households = np.repeat(motifs, X)
                households_ = np.concatenate(households)
                p_genders,p_ages,p_counts = households_[:,0],households_[:,1],households_[:,2]
                p_genders = np.repeat(p_genders, p_counts)
                p_ages    = np.repeat(p_ages, p_counts)

                pids = np.arange(pid_start,pid_start+len(p_genders))
                hids = np.repeat(np.arange(hid_start,hid_start+len(households)), np.repeat(hsize,X))
                zone_persons = np.c_[pids, hids, p_ages, p_genders, np.repeat(zone_id, +len(p_genders))]
                mat_person.append(zone_persons)

                hid_start = hids[-1]+1
                pid_start = pids[-1]+1

        mat_person = np.concatenate(mat_person)
        df_person = pd.DataFrame(mat_person,columns=["pid","hid","age","gender","hzone"])
        df_person.to_csv(self.fn_output, index=False)


if __name__ == '__main__':
    SHG = SubdistrictHouseholdGeneration()
    SHG.run()
