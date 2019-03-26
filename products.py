# This script trains a model and evaluates its performance on the product data
# It could be substantially cleaned up - it's written as a one-off script.

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

_D = 2**15
def simple_one_hot_encoding(s, D=_D):
    s = s.split(" ")
    return {hash(_s) % D : 1 for _s in s}

x = [simple_one_hot_encoding(_x) for _x in x]

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(x,y)

from ftrl import FTRLP

model = FTRLP(D=_D)

model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score as AUC
print('AUC on dairy is %s' % AUC(y_test, model.predict_proba(X_test)[:,1]))


#### Now, do all categories ####

def make_model(data, target):
    x,y = [], []
    for d in data:
        y.append(int(d[-1].rstrip() == target))
        x.append(d[0])
    x = [simple_one_hot_encoding(_x) for _x in x]
    
    ################
    model = FTRLP(D=2**15)
    X_train, X_test, y_train, y_test = tts(x,y)
    model.fit(X_train, y_train)
    _AUC = AUC(y_test, model.predict_proba(X_test)[:,1])
    ###############

    model = FTRLP(D=2**15)
    model.fit(x,y)
    return model, _AUC

labels = ["dairy", "beverages", "beer", "produce", "baking"]

models = [make_model(data, c) for c in labels] 

print(["label : AUC"])
print(["%s : %s" % (l, a[1]) for (l,a) in zip(labels, models)])
     
import numpy as np
def label(row):
    row = row.split(",")
    row[-1] = row[-1].rstrip()
    x = row[0]
    preds = [m[0].predict_one(simple_one_hot_encoding(x))[1] for m in models]
    lab = np.argmax(preds) # use the highest prediction as the predicted label
    row.append(labels[lab]+'\n')
    return ','.join(row)

with open('tmp.dat', encoding="latin-1") as fin:
    with open('labels.csv', 'w') as fout:
        for f in fin:
            out = label(f)
            fout.write(out)














