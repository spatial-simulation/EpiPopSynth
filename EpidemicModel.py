from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy.lib.recfunctions as rfn
import pandas as pd
import numpy as np
import pickle
import os

class EpidemicSimulation:
    SUSC:  int = 0
    INFT:  int = 1
    LATT:  int = 2
    RECV:  int = 3

    HOME:  int = 0
    WORK:  int = 1
    OTHER: int = 2

    def __init__(self, mode, no_experiment):
        path_output = "./Output/Epidemic/Res-%s"%(mode)
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        self.fn_person = "./Output/Populations/%s/pop_%03i.csv"%(mode, (no_experiment%100))
        self.fn_output = "./Output/Epidemic/Res-%s/%s.npy"%(mode, no_experiment)
        self.mode = mode

        self._load_agents()
        self._init_param()

    def _init_param(self):
        # compartment duration
        self.cd_infect = 10
        self.cd_latent = 5

        # simulate days
        self.max_iterday = 100

        # transimissibility
        self.ptrans = 0.04

        # contact setting for different layers
        self.nco  = 1
        self.ncw  = 1
        self.ic_h = 10
        self.ic_w = 2
        self.ic_o = 1

    def _fast_load_structured_array(self, fn_person):
        col_names = ['pids','hids','age']
        tab_person = pd.read_csv(self.fn_person)[col_names].to_numpy()
        p_dtype = np.dtype([("pid", "int32"), ("hid", "int32"), ("age", "int8")])
        tab_person = rfn.unstructured_to_structured(tab_person, p_dtype)
        return tab_person

    def _load_agents(self):
        self.tab_person = self._fast_load_structured_array(self.fn_person)
        self.n_agent = self.tab_person.shape[0]

        # mask = np.ediff1d(self.tab_person["hid"], to_begin=1, to_end=1)
        # indices = np.where(mask)[0]
        # keys = np.unique(self.tab_person["hid"])
        # values = np.fromiter(map(lambda i: self.tab_person["pid"][i[0]:i[1]], zip(indices[:-1],indices[1:])), dtype=list)
        # self.hdict = dict(zip(keys,values))

        df_person = pd.DataFrame(self.tab_person)
        self.hdict = df_person.groupby("hid")["pid"].apply(list).to_dict()
        self.wdict = df_person.groupby("age")["pid"].apply(list).to_dict()

    def _random_binomial_choice(self, pids, prob, replace=False):
        prob = min(prob, 1)
        n_pid = np.random.binomial(len(pids), prob)
        pids_choice = np.random.choice(pids, n_pid, replace=replace)
        return pids_choice

    def init_sim_mat(self):
        person_ids = self.tab_person["pid"]
        this_compartment  = np.repeat(self.SUSC, self.n_agent)
        next_compartment  = np.repeat(self.SUSC, self.n_agent)
        transit_countdown = np.repeat(np.inf,    self.n_agent)
        sim_mat = np.c_[person_ids, this_compartment, next_compartment, transit_countdown]
        s_dtype = np.dtype([("pid", "int32"), ("comp_this", "int8"), ("comp_next", "int8"), ("cd_trans", "float")])
        sim_mat = rfn.unstructured_to_structured(sim_mat, s_dtype)
        return sim_mat

    def simulate_seed(self, sim_mat, n, age=None):
        if age is not None:
            pid = self.tab_person["pid"][np.isin(self.tab_person["age"],age)]
        else:
            pid = self.tab_person["pid"]
        pid_seed = np.random.choice(pid, n, replace=False)

        sim_mat["comp_this"][pid_seed] = self.LATT
        sim_mat["comp_next"][pid_seed] = self.INFT
        sim_mat["cd_trans" ][pid_seed] = self.cd_latent
        return sim_mat

    def simulate_recovery(self, sim_mat):
        pid_recv = sim_mat[(sim_mat["cd_trans"]<=0)*(sim_mat["comp_next"]==self.RECV)]["pid"]
        sim_mat["comp_this"][pid_recv] = self.RECV
        sim_mat["comp_next"][pid_recv] = self.RECV
        sim_mat["cd_trans" ][pid_recv] = np.inf
        return sim_mat,pid_recv

    def simulate_infection(self, sim_mat):
        pid_infectious = sim_mat["pid"][sim_mat["comp_this"]==self.INFT]
        if not pid_infectious.size:
            return sim_mat,np.empty(shape=0,dtype=int)

        # simulate household infection
        hid_infected = self.tab_person["hid"][pid_infectious]
        pid_infected_cand = np.concatenate(list(map(self.hdict.get, hid_infected)))
        pid_infected_h = self._random_binomial_choice(pid_infected_cand, self.ptrans*self.ic_h)

        # simulate work/school infection
        age_infected = self.tab_person["age"][pid_infectious]
        pid_infected_w = []
        for a,c in enumerate(np.bincount(age_infected)):
            if c!=0:
                w_cand = np.random.choice(self.wdict[a], self.ncw*c)
                pid_infected_w.append(self._random_binomial_choice(w_cand, self.ptrans*self.ic_w))
        pid_infected_w = np.concatenate(pid_infected_w)

        # simulate other infection
        n_infect_other = self.nco*len(pid_infectious)
        pid_infected_o = self._random_binomial_choice(sim_mat["pid"], (len(pid_infectious)/self.n_agent)*self.ptrans*self.ic_o)
        pid_infected = np.unique(np.r_[pid_infected_h, pid_infected_w, pid_infected_o])
        sim_mat_infected_ = sim_mat[pid_infected]
        pid_infected = sim_mat_infected_["pid"][sim_mat_infected_["comp_this"]==self.SUSC]

        sim_mat["comp_this"][pid_infected] = self.LATT
        sim_mat["comp_next"][pid_infected] = self.INFT
        sim_mat["cd_trans" ][pid_infected] = self.cd_latent
        return sim_mat,pid_infected

    def simulate_onset(self, sim_mat):
        # simulate latent to infectious
        pid_onset = sim_mat[(sim_mat["cd_trans"]<=0)*(sim_mat["comp_next"]==self.INFT)]["pid"]
        pid_onset = sim_mat[(sim_mat["comp_next"]==self.INFT)]["pid"]

        sim_mat["comp_this"][pid_onset] = self.INFT
        sim_mat["comp_next"][pid_onset] = self.RECV
        sim_mat["cd_trans" ][pid_onset] = self.cd_infect
        return sim_mat

    def run(self):
        sim_mat = self.init_sim_mat()

        sim_mat = self.simulate_seed(sim_mat)

        if self.mode=="mh":
            sim_mat = self.simulate_seed(sim_mat)
        elif self.mode=="di":
            sim_mat = self.simulate_seed(sim_mat)
        else:
            sim_mat = self.simulate_seed(sim_mat)


        stat = []
        for self.sim_date in range(1, self.max_iterday+1):
            sim_mat["cd_trans"] -= 1
            sim_mat,pid_recovered = self.simulate_recovery(sim_mat)
            sim_mat,pid_infected  = self.simulate_infection(sim_mat)
            sim_mat = self.simulate_onset(sim_mat)

            daily_infect_age = np.bincount(self.tab_person["age"][pid_infected])
            daily_infect_age.resize(9)
            stat.append(daily_infect_age)

        stat = np.concatenate(stat).reshape(-1,9)
        np.save(self.fn_output, stat)


if __name__ == '__main__':
    for no_experiment in range(1000):
        print ("Experiment: %s"%(no_experiment))
        for mode in ["mh","ipf","di"]:
            ES = EpidemicSimulation(mode, no_experiment)
            ES.run()