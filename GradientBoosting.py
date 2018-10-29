# import important packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

# read data
## the training dataset has all the model inputs and clustering result in R
## the prediction data set has model inputs and the clustering result in R
input_file = 'traindata.csv'

df = pd.read_csv(input_file)
df.info()

X = df.loc[:, df.columns != 'cluster']
Y = df['cluster']

# use CV to tune the model
params = {'n_estimators':[2000, 2200, 2400, 2600, 2800, 3000],
		'learning_rate': [0.005, 0.01, 0.015, 0.02]}

cv = GridSearchCV(estimator = GradientBoostingClassifier(min_samples_split=2,
														min_samples_leaf=10,
														max_features='auto',
														random_state=0,
														subsample=1),
				param_grid = params,scoring='accuracy',n_jobs=4,iid=False,cv=6).fit(np.array(X), np.array(Y))

print(cv.best_params_)

# use best parameters to fit the model on entire sample 
# in this case n_estimators = 3000, meaning 3000 trees will be built for this model
# learning_rate = 0.01, meaning each new tree will be multiplied with weight 0.01 and added to the model
# higher learning rate means faster learning and potential overfitting
md = GradientBoostingClassifier(learning_rate=0.01,
								n_estimators=3000,
								min_samples_split=2,
								min_samples_leaf=10,
								max_features='auto',
								random_state=10,
								subsample=1).fit(np.array(X), np.array(Y))

# visualize the importance of features
predictors = X.columns
feat_imp = pd.Series(md.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind = 'bar', title = "Importance of Features")
plt.ylabel("Feature Importance Score")
plt.show()

# calculate the model accuracy
pY = pd.DataFrame(md.predict(X))
result = pd.concat([Y, pY], axis = 1)
result.columns = ['cluster', 'predict']
result['err'] = result['cluster'] == result['predict']
print(result.groupby('err').count())

# save the model into pickle format
joblib.dump(md, 'prediction.pkl')
