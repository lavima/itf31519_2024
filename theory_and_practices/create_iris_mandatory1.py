import pandas as pd

dataset =  pd.read_csv('iris.data', header=None)

train = dataset.groupby(4,group_keys=False)[[0,1,2,3,4]].apply(lambda x: x.sample(4))
dataset.drop(train.index)
test = dataset.groupby(4,group_keys=False)[[0,1,2,3,4]].apply(lambda x: x.sample(1))

subset = pd.concat([train, test], ignore_index=True)
print(subset)

subset.to_csv('iris_subset.csv')
