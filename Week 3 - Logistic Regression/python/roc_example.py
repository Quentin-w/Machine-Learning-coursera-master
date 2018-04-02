from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
from ggplot import *

X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, n_informative=5)
Xtrain = X[:9000]
Xtest = X[9000:]
ytrain = y[:9000]
ytest = y[9000:]

clf = LogisticRegression()
clf.fit(Xtrain, ytrain)

preds = clf.predict_proba(Xtest)[:,1]
fpr, tpr, _ = metrics.roc_curve(ytest, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
g = ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')


print(g)

#auc = metrics.auc(fpr,tpr)
#g = ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +\
#    geom_area(alpha=0.2) +\
#    geom_line(aes(y='tpr')) +\
#    ggtitle("ROC Curve w/ AUC=%s" % str(auc))

#g.draw()