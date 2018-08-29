
# import important packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

############################################
#
# Short Quiz
#
############################################

input_file = 'traindata_short.csv'
input_file2 = 'prediction_short.csv'
df = pd.read_csv(input_file)
df2 = pd.read_csv(input_file2)
# print("column name is: " + df.columns)

X = df.loc[:, df.columns != 'cluster_new']
Y = df['cluster_new']
mY = df2['pred_cluster']
# print(type(X))
# print("input is: " + X.columns)

# print(type(Y))
# print("segment is: " + Y.columns)


################ CV ####################
# p_test = {'n_estimators':[1000, 1200, 1400, 1600, 1800, 2000]}
# cv = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.01,min_samples_split=2,min_samples_leaf=10,max_features='auto',random_state=0,subsample = 1),param_grid = p_test,scoring='accuracy',n_jobs=4,iid=False,cv=6).fit(np.array(X), np.array(Y))

# print(cv.best_params_)
# # 2000
# p = cv.best_params_['n_estimators']

# R gbm model specs as follow
# cross.gbm3 <- gbm(cluster_new ~ ., data=df.test2, cv.folds=6, n.trees=2000, shrinkage=0.01, bag.fraction=.75, interaction.depth=5)
# shrinkage
# n.trees
# minimum number of sample to allow splitting node
# minimum leaf size
# maximum criterion, squared error
# random start like seed
# like cv.fold, using subsample for out of bag estimation
# out of bag sample to estimate generalized accuracy
# bootstrap
md = GradientBoostingClassifier(learning_rate=0.01,n_estimators=2000,min_samples_split=2,min_samples_leaf=10,max_features='auto',random_state=10,subsample=0.75).fit(np.array(X), np.array(Y))
predictors = X.columns
feat_imp = pd.Series(md.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind = 'bar', title = "Importance of Features")
plt.ylabel("Feature Importance Score")
plt.show()

pY = pd.DataFrame(md.predict(X))
result = pd.concat([Y, mY, pY], axis = 1)
result.columns = ['cluster','predictR','predictP']
result['err'] = result['cluster'] == result['predictP']
result['diff'] = result['predictR'] == result['predictP']
print(result.groupby('err').count())
# cluster predictR predictP diff
# err 
# False 1891 1891 1891 1891
# True 6109 6109 6109 6109
# 
# Accuracy = 76.3%
print(result.groupby('diff').count())
# cluster predictR predictP err
# diff 
# False 299 299 299 299
# True 7701 7701 7701 7701
#
# Difference between R and Python is 3.7%
# save the model
joblib.dump(md, 'predict_short.pkl')

md2 = joblib.load('predict_short.pkl')
tY = pd.DataFrame(md2.predict(X))
result = pd.concat([result, tY], axis = 1)
result.columns = ['cluster', 'predictR', 'predictP', 'err', 'diff', 'predictT']
result['err2'] = result['cluster'] == result['predictT']
result['diff2'] = result['predictR'] == result['predictT']
result['samemd'] = result['predictP'] == result['predictT']
print(result.groupby('samemd').count())
# cluster predictR predictP err diff predictT err2 diff2
# samemd 
# True 8000 8000 8000 8000 8000 8000 8000 8000
print(result.groupby('diff2').count())


############################################
#
# Long Quiz
#
############################################

input_file = 'traindata_long.csv'
input_file2 = 'prediction_long.csv'
df = pd.read_csv(input_file)
df2 = pd.read_csv(input_file2)
# print("column name is: " + df.columns)

X = df.loc[:, df.columns != 'cluster_new']
Y = df['cluster_new']
mY = df2['pred_cluster']
# print(type(X))
# print("input is: " + X.columns)

# print(type(Y))
# print("segment is: " + Y.columns)

################ CV ####################
# p_test = {'n_estimators':[2000, 2200, 2400, 2600, 2800, 3000]}
# cv = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.01,min_samples_split=2,min_samples_leaf=10,max_features='auto',random_state=0,subsample=1),param_grid = p_test,scoring='accuracy',n_jobs=4,iid=False,cv=6).fit(np.array(X), np.array(Y))

# print(cv.best_params_)
# p = cv.best_params_['n_estimators']


# R gbm model specs as follow
# cross.gbm4 <- gbm(cluster_new ~ ., data=df.test, cv.folds=6, n.trees=3000, shrinkage=0.01, bag.fraction=.75, interaction.depth=5)
# shrinkage
# n.trees
# minimum number of sample to allow splitting node
# minimum leaf size
# maximum criterion, squared error
# random start like seed
# like cv.fold, using subsample for out of bag estimation
# out of bag sample to estimate generalized accuracy
# bootstrap
md = GradientBoostingClassifier(learning_rate=0.01,n_estimators=3000,min_samples_split=2,min_samples_leaf=10,max_features='auto',random_state=10,subsample=0.75).fit(np.array(X), np.array(Y))
predictors = X.columns
feat_imp = pd.Series(md.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind = 'bar', title = "Importance of Features")
plt.ylabel("Feature Importance Score")
plt.show()

pY = pd.DataFrame(md.predict(X))
result = pd.concat([Y, mY, pY], axis = 1)
result.columns = ['cluster','predictR','predictP']
result['err'] = result['cluster'] == result['predictP']
result['diff'] = result['predictR'] == result['predictP']
print(result.groupby('err').count())
# cluster predictR predictP diff
# err 
# False 655 655 655 655
# True 7345 7345 7345 7345
# 
# Accuracy = 91.8%
print(result.groupby('diff').count())
# cluster predictR predictP err
# diff 
# False 166 166 166 166
# True 7834 7834 7834 7834
#
# Difference between R and Python is 2%
# save the model
joblib.dump(md, 'predict_long.pkl')

md2 = joblib.load('predict_long.pkl')
tY = pd.DataFrame(md2.predict(X))
result = pd.concat([result, tY], axis = 1)
result.columns = ['cluster', 'predictR', 'predictP', 'err', 'diff', 'predictT']
result['err2'] = result['cluster'] == result['predictT']
result['diff2'] = result['predictR'] == result['predictT']
result['samemd'] = result['predictP'] == result['predictT']
print(result.groupby('samemd').count())
print(result.groupby('diff2').count())
# cluster predictR predictP err diff predictT err2 diff2
# samemd 
# True 8000 8000 8000 8000 8000 8000 8000 8000
# cluster predictR predictP err diff predictT err2 samemd
# diff2 
# False 166 166 166 166 166 166 166 166
# True 7834 7834 7834 7834 7834 7834 7834 7834