import joblib

model = joblib.load("explore_randomforest_classification_2.pkl")
print("Jumlah fitur:", model.n_features_in_)
print(model.feature_names_in_)