import numpy as np 
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Read data
train_data = pd.read_csv('AggYears.csv')
test_data = pd.read_csv('22sea.csv')

# Pre processing
train_data["Team"] = train_data["Team"].astype(str)
test_data["Team"] = test_data["Team"].astype(str)

train_data["Age"] = train_data["Age"].astype(str)
test_data["Age"] = test_data["Age"].astype(str)

train_data["MOV"] = train_data["MOV"].astype(str)
test_data["MOV"] = test_data["MOV"].astype(str)

train_data["SOS"] = train_data["SOS"].astype(str)
test_data["SOS"] = test_data["SOS"].astype(str)

train_data["SRS"] = train_data["SRS"].astype(str)
test_data["SRS"] = test_data["SRS"].astype(str)

train_data["ORtg"] = train_data["ORtg"].astype(str)
test_data["ORtg"] = test_data["ORtg"].astype(str)

train_data["DRtg"] = train_data["DRtg"].astype(str)
test_data["DRtg"] = test_data["DRtg"].astype(str)

train_data["NRtg"] = train_data["NRtg"].astype(str)
test_data["NRtg"] = test_data["NRtg"].astype(str)

train_data["Pace"] = train_data["Pace"].astype(str)
test_data["Pace"] = test_data["Pace"].astype(str)

train_data["FTr"] = train_data["FTr"].astype(str)
test_data["FTr"] = test_data["FTr"].astype(str)

train_data["3PAr"] = train_data["3PAr"].astype(str)
test_data["3PAr"] = test_data["3PAr"].astype(str)

train_data["TS%"] = train_data["TS%"].astype(str)
test_data["TS%"] = test_data["TS%"].astype(str)

train_data["eFG%"] = train_data["eFG%"].astype(str)
test_data["eFG%"] = test_data["eFG%"].astype(str)

train_data["TOV%"] = train_data["TOV%"].astype(str)
test_data["TOV%"] = test_data["TOV%"].astype(str)

train_data["ORB%"] = train_data["ORB%"].astype(str)
test_data["ORB%"] = test_data["ORB%"].astype(str)

train_data["FT/FGA"] = train_data["FT/FGA"].astype(str)
test_data["FT/FGA"] = test_data["FT/FGA"].astype(str)

train_data["DeFG%"] = train_data["DeFG%"].astype(str)
test_data["DeFG%"] = test_data["DeFG%"].astype(str)

train_data["DTOV%"] = train_data["DTOV%"].astype(str)
test_data["DTOV%"] = test_data["DTOV%"].astype(str)

train_data["DRB%"] = train_data["DRB%"].astype(str)
test_data["DRB%"] = test_data["DRB%"].astype(str)

train_data["DFT/FGA"] = train_data["DFT/FGA"].astype(str)
test_data["DFT/FGA"] = test_data["DFT/FGA"].astype(str)

train_data.info()
test_data.info()

# Pre processing

# Select features
X = train_data.drop(['PFWins', 'Team', 'Rk', 'Year', 'Age'], axis=1)
y = train_data['PFWins']
test_data_set = test_data.drop(['Team', 'Rk', 'Year', 'Age'], axis=1)

# List how many labels we are using for training
cat_features = list(range(0, X.shape[1]))
print(cat_features)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Implement CatBoostRegressor
clf = CatBoostRegressor(
    task_type="GPU",
    loss_function='RMSE',
    learning_rate=.01,
    iterations=1100,
    boosting_type='Ordered',
    depth=8,
    bootstrap_type='Bayesian',
    silent=True,
)

clf.fit(X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        # verbose=False,
    )

tested_data = clf.predict(test_data_set)
print("Playoff Wins for each team:")
for i in range(len(tested_data)):
    print(test_data["Team"][i],",",tested_data[i])
print(clf.score(X_train, y_train))
print(mean_squared_error(y_test, clf.predict(X_test)))
# print feature names with corresponding feature importance
print(dict(zip(X_train.columns, clf.feature_importances_)))
