from sklearn import datasets as ds
from sklearn import tree
from sklearn.model_selection import cross_val_score
import pydotplus


boston = ds.load_boston()
dtree = tree.DecisionTreeRegressor()
dtree.fit(boston.data[0:20], boston.target[0:20])
target_predict = dtree.predict(boston.data[11:20])
#cross value score
print(cross_val_score(dtree, boston.data, boston.target))
with open("boston.dot", "w") as f:
    f = tree.export_graphviz(dtree, out_file=f)
dot_data = tree.export_graphviz(dtree, out_file=None,
                                filled=True, rounded=True,
                                feature_names=boston.feature_names,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("boston.pdf")