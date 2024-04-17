import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import partial_dependence
from pathlib import Path

file_dir = Path(__file__).resolve().parents[0]

# Load input out output data
X = pd.read_csv(file_dir / "res//trn//list_of_inputs.csv")
# X = X2[X2['design_case']=='ASHP']
del X["flow_rate"], X["inf"] #,  X['design_case']
X_cols = X.columns
y = pd.read_csv(file_dir / "res//sim_results.csv")
y['total_costs_0'] = y['el_bill_0']+y['gas_bill']
y['total_emission'] = y['el_em']+y['gas_em']

# Merge
data = pd.merge(X, y, on="label")
# data = data[data["draw"].str.contains("old") == True]
# removing the data that does not contribute to flow factor
# data = data[data['design_case'].str.contains('ASHP')==False]
# data = data[data['design_case'].str.contains('cp_PV')==False]

# data = data[data['design_case'].str.contains('ST')==True]

#These two lines for obtaining only new values, of 1 indulation level
# data = data[data["draw"].str.contains("old") == False]
# data = data[data['r_level']=='r0']
#%%
# Select target variable
y_col = "total_costs_0"
y = data[y_col]
X = data[X_cols].copy()
del X["label"]
# Ordinal encoding
for col in X.select_dtypes(include="object").columns:
    enc = OrdinalEncoder()
    X[col] = enc.fit_transform(X[col].values.reshape(-1, 1))
#%%
model = GradientBoostingRegressor()
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
scores2 = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"Mean MAE: {np.mean(scores)}" + f"\t R2: {np.mean(scores2)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
# model.fit(X, y)

#%%
plot_title = data['design_case'].iloc[0]
# for col in X.columns:
x_col = "coll_area"
unq_vals = np.unique(X[x_col])
ice_curves = np.zeros((len(X), len(unq_vals)))
for k, val in enumerate(unq_vals):
    X_tmp = X.copy()
    X_tmp[x_col] = val
    y_pred = model.predict(X_tmp)
    ice_curves[:, k] = y_pred

pd_curve = np.mean(ice_curves, axis=0)
# Plot ICE and PD curve
plt.figure(dpi=120)
plt.plot(unq_vals, pd_curve, "r-", label="Average")
plt.plot(unq_vals, ice_curves.T, "k-", alpha=0.1, linewidth=0.2)
plt.legend()
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(plot_title)
plt.ylim([1000,4500])
plt.show()

#%% c-ICE curves
unq_vals = np.unique(X[x_col])
# c_ice_curves = np.zeros((len(X), len(unq_vals)))

# 1. Compute the reference value (median of "volume" in this case)
ref_value = np.min(X[x_col])

# 2. Compute predictions for the dataset at the reference value
X_ref = X.copy()
X_ref[x_col] = ref_value
y_ref_pred = model.predict(X_ref)

# 3. Center the ICE curves by subtracting reference predictions
# Ensure that y_ref_pred is broadcasted correctly across rows
c_ice_curves = ice_curves - y_ref_pred[:, np.newaxis]

#%% plot_ice cruves
pd_c_curve = np.mean(c_ice_curves, axis=0)
# Plot ICE and PD curve
plt.figure(dpi=120)
plt.plot(unq_vals, pd_c_curve, "r-", label="Average")
plt.plot(unq_vals, c_ice_curves.T, "k-", alpha=0.1, linewidth=0.2)
plt.legend()
plt.xlabel(x_col)
plt.ylabel(f"Savings in operational costs [EUR]")
plt.title(plot_title)
plt.ylim([None,None])
plt.show()
#%% Scatter plot of 1 parameter vs second parameter, color denotes KPI
# kpi = 'total_costs_0'
# x_param = 'r_level'
# y_param = 'coll_area'

# fig, ax = plt.subplots()
# # scatter = ax.scatter(data[x_param], data[y_param], edgecolors=plt.cm.viridis(data[kpi]), facecolors='none')
# scatter = ax.scatter(data[x_param], data[y_param], c = data[kpi], facecolors='none', s=data['label'])

# cbar = fig.colorbar(scatter, ax=ax)
# cbar.set_label(kpi)
# plt.xlabel(x_param)
# plt.ylabel(y_param)
# plt.show()