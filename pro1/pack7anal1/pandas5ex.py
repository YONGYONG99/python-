import pandas as pd

# 문제 3
# 1)
df = pd.read_csv("../testdata/titanic_data.csv" , header = None)
print(df[[5]])
Age = df[5].drop(0).fillna(0)
print(Age)
# print(df)
cut = [1,20,35,60,150]
# result_cut = pd.cut(Age, cut)
# print(result_cut)