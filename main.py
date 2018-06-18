import pandas as pd

data = pd.read_csv('./dataset/csv/train.csv', encoding='utf-8', delimiter='\t')
data = data.fillna(method="ffill")
print(data.tail(50))
#
# train_dataset = dataset.Dataset("./dataset/raw/valid.txt", "./dataset/csv/valid", debug=False).create()
