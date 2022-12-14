# %%
import pandas as pd
data = {
    "x1": [1,2,1,3,4,3,2,10,11,12,11,10,13,12],
    "x2": [2,2,6,1,3,4,5,10,10,11,12,13,12,14],
    "c": [0,0,0,0,0,0,0,1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)  
df.plot(
    x="x1",
    y="x2",
    kind="scatter",  
    title="Datenset",  
    figsize=(10,10),  
    c="c",  
    cmap='winter'
)

# %% Logistic Regression
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

log_regressor = LogisticRegression(solver="lbfgs")  
log_regressor.fit(df[['x1', 'x2']], df[['c']].to_numpy().ravel())
log_prediciton = log_regressor.predict(df[['x1','x2']])


cm = confusion_matrix(  
    y_true=df[['c']].to_numpy().ravel(),  
    y_pred=log_prediciton
)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])  
disp.plot()
# %%
from sklearn.tree import DecisionTreeClassifier

tree_classifyer = DecisionTreeClassifier()  
tree_classifyer.fit(df[['x1', 'x2']], df[['c']].to_numpy().ravel())
tree_prediction = tree_classifyer.predict(df[['x1', 'x2']])

tree_cm = confusion_matrix(
    y_true=df[['c']].to_numpy().ravel(),  
    y_pred=tree_prediction
)
disp = ConfusionMatrixDisplay(confusion_matrix=tree_cm, display_labels=["0", "1"])
disp.plot()  
# %%
from sklearn import tree
tree.plot_tree(tree_classifyer)
# %%
