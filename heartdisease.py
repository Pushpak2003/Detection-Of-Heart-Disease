import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#load dataset
df = pd.read_csv("heart_deasease.csv")
df.head(6)
df.shape

#Feature enginnering
df.isnull().sum()

#plotting graph
plt.figure(figsize=(20,20))
ax = sns.boxplot(data=df)

#importing statistcs
from scipy import stats
z = np.abs(stats.zscore(df))
print(z)

threshold = 3
print(np.where(z > 3))# The first array contains the list of row numbers and second array respective column numbers

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[(z < 3).all(axis=1)]
df.shape

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape

plt.figure(figsize=(20,20))
ax = sns.boxplot(data=df)

#feature selection
plt.figure(figsize=(20,20))
d = sns.heatmap(df.corr(),cmap="coolwarm",annot= True)

# df = df.drop(columns= "chol")
# df.head()

df.describe()

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])# creating dummy variable
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] # we have taken these columns for scale down
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

dataset.head()

dataset.tail()

dataset.describe()

#Data visualisation
sns.pairplot(df , hue="target", height=3, aspect=1);

#model selection
y = dataset['target']
X = dataset.drop(['target'], axis = 1) 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)  

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#K-Nearest Neighbour 
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X_train, y_train)
score=cross_val_score(knn_classifier,X_train,y_train,cv=10)
y_pred_knn = knn_classifier.predict(X_test)
print(accuracy_score(y_test, y_pred_knn))
print(score.mean())
knn_classifier  = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
 metric_params=None, n_jobs=1, n_neighbors=5, p=1,
 weights='uniform')
knn_classifier.fit(X_train, y_train)
score=cross_val_score(knn_classifier,X_train,y_train,cv=10)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)
score=cross_val_score(knn_classifier,X_train,y_train,cv=10)
score.mean()

#Confusion matrix
cm = confusion_matrix(y_test, y_pred_knn)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()

print(classification_report(y_test, y_pred_knn))

#random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred_rf)

score=cross_val_score(rf_classifier,X_train,y_train,cv=10)
score.mean()

#XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
       learning_rate=0.1, max_delta_step=0, max_depth=15,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       objective='binary:logistic', random_state=23,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       subsample=1, verbosity=1)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
print(accuracy_score(y_test, y_pred_xgb))

score=cross_val_score(xgb_classifier,X_train,y_train,cv=10)
score.mean()

#AdaBoost with Random Forest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
ada_clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100), n_estimators=100)
ada_clf.fit(X_train, y_train)

y_pred_adb = ada_clf.predict(X_test)
accuracy_score(y_test, y_pred_adb)

score=cross_val_score(ada_clf,X_train,y_train,cv=10)
score.mean()

#Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train)

y_pred_adb = gbc_clf.predict(X_test)
accuracy_score(y_test, y_pred_adb)

score=cross_val_score(gbc_clf,X_train,y_train,cv=10)
score.mean()

# Save Model
## Pickle
from xgboost import XGBClassifier
import pickle

# save model
pickle.dump(knn_classifier, open('model.pkl', 'wb'))

# load model
Heart_disease_detector_model = pickle.load(open('model.pkl', 'rb'))

# predict the output
y_pred = Heart_disease_detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of K – Nearest Neighbor model: \n',confusion_matrix(y_test, y_pred),'\n')

# show the accuracy
print('Accuracy of K – Nearest Neighbor  model = ',accuracy_score(y_test, y_pred))