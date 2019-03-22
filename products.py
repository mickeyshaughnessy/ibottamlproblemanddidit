# This script trains a model and evaluates it's performance on the product data

data = []
with open('labeled.csv') as f:
    for line in f:
        line = line.split(',')
        if len(line) == 4:
            data.append(line)
header = data[0]
del data[0]

#focus in on dairy
x,y = [], []
for d in data:
    y.append(int(d[-1].rstrip() == "dairy"))
    x.append(d[0])

def simple_one_hot_encoding(s, D=2**15):
    s = s.split(" ")
    return {hash(_s) % D : 1 for _s in s}


x = [simple_one_hot_encoding(_x) for _x in x]

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(x,y)

from ftrl import FTRLP

model = FTRLP(D=2**15)

model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score as AUC

print(AUC(y_test, model.predict_proba(X_test)[:,1]))






