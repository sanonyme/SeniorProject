import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")



import pickle

df = pd.read_csv("Crop_recommendation.csv")


c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.label #Target.
X=df[['N','P','K','temperature','humidity','ph','rainfall']] #Fields used in ML algo.



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)


# ## Pre processing

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)


# # Sample data

# Select random values in the dataset to test predictions below
data = df.iloc[1000]
print(data[7])
print('\n')

# Slice the features for prediction
input_test = data[:-2]
print(input_test)
print('\n')

# Format the data to use in models
prediction_data = input_test.to_numpy().reshape(1, -1)
scaled_prediction_data = min_max_scaler.transform(prediction_data)
print(prediction_data)
print(scaled_prediction_data)


# ## KNN


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)


knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train_scaled, y_train)

from sklearn.metrics import classification_report
knnPredictor = knn.predict(X_test_scaled)

knn.predict(scaled_prediction_data)


from sklearn.svm import SVC

svc_linear = SVC(kernel='linear').fit(X_train_scaled, y_train)
svc_rbf = SVC(kernel='rbf').fit(X_train_scaled, y_train)
svc_poly = SVC(kernel='poly').fit(X_train_scaled, y_train)


svc_linear.predict(scaled_prediction_data)


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)


dt.predict(prediction_data)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42).fit(X_train, y_train)

clf.predict(prediction_data)


from sklearn.ensemble import GradientBoostingClassifier
gboost = GradientBoostingClassifier().fit(X_train, y_train)


gboost.predict(prediction_data)


from sklearn.ensemble import VotingClassifier

ensemble_classifier_h = VotingClassifier(estimators= [
    ('knn', knn), 
    ('svc_linear', svc_linear), 
    ('svc_poly', svc_poly), 
    ('svc_rbf', svc_rbf), 
    ('dt', dt), 
    ('rf', clf), 
    ('xgboost', gboost)
    ],
    voting='hard')

ensemble_classifier_h = ensemble_classifier_h.fit(X,y)

ensemble_classifier_h.score(X, y)

ensemble_classifier_h.predict(prediction_data)



def crop_prediction(x):
    return ensemble_classifier_h.predict(x)


print("Hello")
print(crop_prediction(prediction_data))


pickle.dump(ensemble_classifier_h, open("model.pkl", "wb"))