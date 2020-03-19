import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle


def GridSearchRF(X_train,y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    # use a full grid over all parameters
    param_grid = {"max_depth": [4,5,7, None],
                "n_estimators":[100,200,300,400,500],
                "max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "min_samples_leaf": [1, 3, 10],
                "bootstrap": [True, False]}
                #"criterion": ["gini", "entropy"]}

    forest_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                    param_grid = param_grid,
                   # criterion='gini',
                    scoring="accuracy",  #metrics
                    cv = 3,            #cross-validation
                    n_jobs = -1)          #number of core

    forest_grid.fit(X_train,y_train) #fit

    forest_grid_best = forest_grid.best_estimator_ #best estimator
    print("Best Model Parameter: ",forest_grid.best_params_)



df = pd.read_csv(r"path.csv", header=0,index_col=0,parse_dates=True)

X_train, X_test, y_train, y_test = train_test_split(df[df.columns[df.columns != 'target']], df.target, test_size=0.2,shuffle=False)

print('train size:', X_train.shape[0])
print('test size:', X_test.shape[0])
print('train data:', X_train)
print('train target:', y_train)


#GridSearchRF(X_train,y_train)

#モデルの訓練
model = RandomForestClassifier(n_estimators=200,random_state=0, n_jobs=-1)
model.fit(X_train, y_train)

#モデルの保存
pickle.dump(model, open( r'finalized_model.sav', 'wb'))

#モデルの学習結果
print(classification_report(y_test,model.predict(X_test)))

test_x = X_train.iloc[[10]]
print(model.predict(test_x))
print(model.predict_proba(test_x))


import numpy as np
import matplotlib.pyplot as plt
#特徴量の重要度
feature = model.feature_importances_
#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]
#特徴量の名前
label = df.columns[0:]

plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), label[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()