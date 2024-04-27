import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("ufc-master.csv/ufc-master.csv")
not_include_list = ["R_odds","B_odds","R_ev","B_ev","r_dec_odds","b_dec_odds","r_sub_odds","b_sub_odds","r_ko_odds","b_ko_odds","date","location","country","R_fighter","B_fighter","finish","finish_details","finish_round_time","total_fight_time_secs"]
processed_df = df.drop(columns=not_include_list)
X = processed_df.drop(columns=["Winner"])
X = X.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
string_cols = X.select_dtypes(['object']).columns

# one-hot encode all string columns
new_df = pd.get_dummies(X, columns=string_cols, drop_first=True)

X = new_df

y = df["Winner"]
# select all boolean columns
bool_cols = new_df.select_dtypes(['bool']).columns

# convert boolean values to 0/1
new_df[bool_cols] = new_df[bool_cols].astype(int)

from sklearn.preprocessing import LabelEncoder

# assume 'df' is your dataframe and 'column_name' is the column you want to label encode
le = LabelEncoder()

y = le.fit_transform(y)

nan_cols = X.columns[X.isna().any()].tolist()

for col in nan_cols:
    X[f'is_nan_{col}'] = X[col].isna().astype(int)
    X[col] = X[col].fillna(0)

from sklearn.decomposition import PCA


pca = PCA(n_components=0.95)  # retain 95% of the variance

# Fit the PCA object to the data and transform it
X_pca = pca.fit_transform(X)

# Get the most important columns (features)
importance = pca.components_[0]

# Get the feature names
feature_names = X.columns

# Sort the feature names by importance
sorted_features = sorted(zip(feature_names, importance), key=lambda x: abs(x[1]), reverse=True)

# Print the top 10 most important features
print("Top 10 most important features:")
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance:.4f}")


import matplotlib.pyplot as plt

# Get the top 10 most important features
# sorted_features = sorted(zip(feature_names, importance), key=lambda x: abs(x[1]), reverse=True)

# Extract the feature names and importance values
feature_names = [x[0] for x in sorted_features[:10]]
importance_values = [x[1] for x in sorted_features[:10]]

# Create a bar graph
plt.barh(range(len(feature_names)), importance_values)
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features')
plt.show()

include_list = ["B_avg_SIG_STR_landed",
"B_avg_SIG_STR_pct",
"B_avg_SUB_ATT",
"B_avg_TD_landed",
"B_avg_TD_pct",
"B_total_rounds_fought",
"B_Height_cms",
"B_Reach_cms",
"R_avg_SIG_STR_landed",
"R_avg_SIG_STR_pct",
"R_avg_SUB_ATT",
"R_avg_TD_landed",
"R_avg_TD_pct",
"R_Height_cms",
"R_Reach_cms",
"R_age",
"B_age",
"lose_streak_dif",
"win_streak_dif",
"longest_win_streak_dif",
"win_dif",
"loss_dif",
"total_round_dif",
"total_title_bout_dif",
"ko_dif",
"sub_dif",
"height_dif",
"reach_dif",
"age_dif",
"sig_str_dif",
"avg_sub_att_dif",
"avg_td_dif"]

X   = X[include_list]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

pca = PCA(n_components=0.95)  # retain 95% of the variance

# Fit the PCA object to the data and transform it
X_pca = pca.fit_transform(X)

# Get the most important columns (features)
importance = pca.components_[0]

# Get the feature names
feature_names = X.columns
sorted_features = sorted(zip(feature_names, importance), key=lambda x: abs(x[1]), reverse=True)

# Print the top 10 most important features
print("Top 10 most important features:")
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance:.4f}")


import matplotlib.pyplot as plt

# Get the top 10 most important features
# sorted_features = sorted(zip(feature_names, importance), key=lambda x: abs(x[1]), reverse=True)

# Extract the feature names and importance values
feature_names = [x[0] for x in sorted_features[:10]]
importance_values = [x[1] for x in sorted_features[:10]]

# Create a bar graph
plt.barh(range(len(feature_names)), importance_values)
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features')
plt.show()