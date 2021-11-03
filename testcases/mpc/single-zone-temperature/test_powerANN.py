import joblib

power = joblib.load("powerANN.pkl")

# examine the cv results
print(power.cv_results_)

# examine best scores
print(power.best_estimator_)
print(power.best_score_)