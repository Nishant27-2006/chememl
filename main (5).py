
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('all_smiles_data.csv')

# Preprocessing
data_cleaned = data.dropna(subset=['activity'])
data_cleaned['pubchem_smiles_length'] = data_cleaned['pubchem_smiles_cleaned'].apply(lambda x: len(str(x)))
data_cleaned['alogps_smiles_length'] = data_cleaned['alogps_smiles_cleaned'].apply(lambda x: len(str(x)))

# Features and target
X = data_cleaned[['pubchem_smiles_length', 'alogps_smiles_length']]
y = data_cleaned['activity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

nn_model = MLPRegressor(max_iter=500)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

# Evaluate models
rf_mse = mean_squared_error(y_test, y_pred_rf)
dt_mse = mean_squared_error(y_test, y_pred_dt)
nn_mse = mean_squared_error(y_test, y_pred_nn)

# Print results
print(f"Random Forest MSE: {rf_mse}")
print(f"Decision Tree MSE: {dt_mse}")
print(f"Neural Network MSE: {nn_mse}")
