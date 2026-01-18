
from collections import Counter
from operator import itemgetter
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy import interpolate

import pandas as pd
import numpy as np


config = {
    "font.family": "serif",
    # "font.serif": ["SimHei"],
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
}
plt.rcParams.update(config)


class PlotMotifDistribution:
    def __init__(self):
        self.fn_survey = r"./cfps2018crossyearid_202104.dta"
        self.age_bounds = np.array([10,20,30,40,50,60,70,80])
        self.max_hsize = 6

    def get_motif(self, subpop, subname):
        household_list = []
        for fid,household in subpop.groupby("fid18"):
            if not household.size:
                continue

            household_structure = Counter(map(tuple,household[["gender","age_groups"]].values.tolist()))
            household_structure = sorted(household_structure.items(), key=itemgetter(0))
            household_list.append(tuple(household_structure))
        household_count = Counter(household_list)

        values = sorted(household_count.values(), reverse=True)
        names = range(1, len(values)+1)

        x_log = np.log10(names).reshape(-1,1)
        y_log = np.log10(values)

        model = linear_model.LinearRegression()
        model.fit(x_log,y_log)
        coef = model.coef_
        model_intercept = model.intercept_

        r2 = model.score(x_log,y_log)
        fit_x0,fit_x1 = min(x_log),max(x_log)
        fit_y0,fit_y1 = model.predict([fit_x0]),model.predict([fit_x1])
        textstr = "$\mathregular{y=%0.2f*x+%0.2f}$ \n $\mathregular{r^2=%0.3f}$"%(coef, model_intercept, r2)
        [fit_x0,fit_x1,fit_y0,fit_y1] = map(lambda x:10**x, [fit_x0,fit_x1,fit_y0,fit_y1])

        fig, ax = plt.subplots(figsize=(8,6))
        ax.loglog()
        ax.scatter(names,values, s=20, color="None", edgecolor="orangered", marker="o", label="Observation", zorder=12)
        ax.plot([fit_x0,fit_x1], [fit_y0,fit_y1], color="royalblue", lw=2, label="Linear Regression", zorder=11)

        ax.text(0.012, 0.017, textstr, transform=ax.transAxes, fontsize=20,
                verticalalignment='bottom', bbox=dict(boxstyle='Square', facecolor='white', alpha=0.8))

        ax.legend(fontsize=18)
        ax.set_ylabel("Probability Density Function (Log Scale)",   fontsize=18, labelpad=2,  weight='bold' )
        ax.set_xlabel("Rank of Household Structures (Log Scale)", fontsize=18, weight='bold')

        ax.grid(linestyle='-.', alpha=0.6, lw=0.8)
        # plt.tight_layout()
        plt.savefig(r"./%s.svg"%(subname), format="svg")
        # plt.show()





    def run(self):
        # with open(self.fn_survey, mode="rb") as fh:
        #     df = pd.read_stata(fh, convert_categoricals=True)

        # df = df[["psu", "pid", "fid18", "gender", "birthy", "selfrpt18", "subpopulation", "subsample","inroster16"]]
        # df = df.loc[df["fid18"].apply(np.isreal)]
        # df = df.loc[df.fid18!="不适用"]
        # df = df.loc[np.isin(df["gender"], ["男","女"])]
        # df["gender"] = (df["gender"]=="男").astype(int)
        # df.to_excel("./person_survey.xls")
        # exit()

        df = pd.read_excel("./person_survey.xls")
        df = df.loc[~df.birthy.isnull()]
        df["age"] = (2022-df["birthy"].astype('int'))
        df["age_groups"] = np.searchsorted(self.age_bounds, df["age"], side="right")
        # df["fid18"] = df["fid18"].astype(int)

        for subname,subpop in df.groupby("subpopulation"):
            # if subname in ["上海市子总体"]:
            #     continue

            print (subname)
            self.get_motif(subpop, subname)





if __name__ == '__main__':
    PMD = PlotMotifDistribution()
    PMD.run()