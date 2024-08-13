"""
This file contains the code in order to tune each of the 
gradient boosted regressors

Add to the notebook in order to tune
"""

# def objective(trial):
#     param = {
#         "verbosity": 0,
#         "objective": "reg:squarederror",
#         "n_estimators" : trial.suggest_int("n_estimators", 70, 270, step=20),
#         # use exact for small dataset.
#         "tree_method": "exact",
#         # defines booster, gblinear for linear functions.
#         "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear"]),
#         # L2 regularization weight.
#         "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
#         # L1 regularization weight.
#         "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
#         # sampling ratio for training data.
#         "subsample": trial.suggest_float("subsample", 0.2, 1.0),
#         # sampling according to each tree.
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
#     }

#     if param["booster"] in ["gbtree", "dart"]:
#         # maximum depth of the tree, signifies complexity of the tree.
#         param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
#         # minimum child weight, larger the term more conservative the tree.
#         param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
#         param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
#         # defines how selective algorithm is.
#         param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
#         param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

#     if param["booster"] == "dart":
#         param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
#         param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
#         param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
#         param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

#     model = XGBBoost(
#         num_targets=6,
#         **param
#     )
#     model.fit(
#         X_train=full_training_data.csv_aug.iloc[train_data.indices],
#         Y_train=full_training_data.plant.labels.iloc[train_data.indices],
#         X_val=full_val_data.csv_aug.iloc[val_data.indices],
#         Y_val=full_val_data.plant.labels.iloc[val_data.indices]
#     )

#     predictions = model.predict(full_val_data.csv_aug.iloc[val_data.indices])
#     return r2_score(full_val_data.plant.labels.iloc[val_data.indices], predictions)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)


# from sklearn.metrics import mean_squared_error, root_mean_squared_error
# import optuna

# def objective(trial):
#     params = {
#         "iterations": 2000,
#         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
#         "depth": trial.suggest_int("depth", 1, 10),
#         "subsample": trial.suggest_float("subsample", 0.05, 1.0),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
#     }

#     model = CatBoost(
#         num_targets=6,
#         **params
#     )
#     model.fit(
#         X_train=full_training_data.csv_aug.iloc[train_data.indices],
#         Y_train=full_training_data.plant.labels.iloc[train_data.indices],
#         X_val=full_val_data.csv_aug.iloc[val_data.indices],
#         Y_val=full_val_data.plant.labels.iloc[val_data.indices]
#     )

#     predictions = model.predict(full_val_data.csv_aug.iloc[val_data.indices])
#     return r2_score(full_val_data.plant.labels.iloc[val_data.indices], predictions)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)


# import optuna

# def objective(trial):
#     params = {
#         "objective": "regression",
#         "metric": "rmse",
#         "n_estimators": 1000,
#         "verbosity": -1,
#         "bagging_freq": 1,
#         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
#         "subsample": trial.suggest_float("subsample", 0.05, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
#         "verbosity": -1,
#     }

#     model = LightGBoost(
#         num_targets=6,
#         **params
#     )
#     model.fit(
#         X_train=full_training_data.csv_aug.iloc[train_data.indices],
#         Y_train=full_training_data.plant.labels.iloc[train_data.indices],
#         X_val=full_val_data.csv_aug.iloc[val_data.indices],
#         Y_val=full_val_data.plant.labels.iloc[val_data.indices]
#     )

#     predictions = model.predict(full_val_data.csv_aug.iloc[val_data.indices])
#     return r2_score(full_val_data.plant.labels.iloc[val_data.indices], predictions)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)
