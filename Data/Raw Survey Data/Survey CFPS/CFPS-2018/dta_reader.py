import pandas as pd
import numpy as np

from collections import Counter


# fn = r"./cfps2018famecon_202101.dta"
fn = r"./cfps2018crossyearid_202104.dta"


with open(fn, mode="rb") as fh:
    df = pd.read_stata(fh, convert_categoricals=True)

print (np.unique(df.subpopulation))
exit()



# df = df[['psu', 'pid', 'fid18', 'gender', 'birthy', 'selfrpt18', 'subpopulation', 'subsample','inroster16']]
# # df = df.loc[df.inroster16=='不适用']
# df = df.loc[df["subpopulation"]=="广东省子总体"]
# df = df.dropna(axis=0)
# df.to_excel("./cfps_2018.xls")

df = pd.read_excel("./cfps_2018.xls")
df = df[df['fid18'].apply(lambda x: str(x).isdigit())]

# df["fsize"] = df.groupby(["psu","fid18"])["pid"].transform(lambda x: len(np.unique(x)))
df["fsize"] = df.groupby(["psu","fid18"])["pid"].transform(len)


print (df.loc[df.fsize>=17].sort_values("birthy"))


# print (np.bincount(df.fsize))








# df = df[df['fid18'].apply(lambda x: str(x).isdigit())]
# df = df[df['fid18'].apply(lambda x: str(x))]

# df["fid18"] = df["fid18"].apply(int)


# df["fid18"] = df["fid18"].apply(lambda x: int(x) if x.isdecimal else -9999)
# df["fid18"] = df.loc[df["fid18"]>=0]

# print (df.loc[df["fid18"].str.isdecimal()])
# print (df["fid18"].apply(type))


# df["fsize"] = df.groupby(["psu","fid18"])["pid"].transform(lambda x: len(np.unique(x)))

# print (np.bincount(df.fsize))
# print (df.fid18)

# for i,(fid,gp) in enumerate(df.groupby("fid18")):
#     print (fid, gp)

#     if i>10:
#         break



