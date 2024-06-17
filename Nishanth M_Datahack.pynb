import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

train_features = pd.read_csv('/content/training_set_features.csv')
train_labels = pd.read_csv('/content/training_set_labels.csv')
test_features = pd.read_csv('/content/test_set_features.csv')
submission_format = pd.read_csv('/content/submission_format.csv')


numeric_features = ['xyz_concern', 'xyz_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands', 'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face', 'doctor_recc_xyz', 'doctor_recc_seasonal', 'chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance', 'opinion_xyz_vacc_effective', 'opinion_xyz_risk', 'opinion_xyz_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'household_adults', 'household_children']

categorical_features = ['age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', MultiOutputClassifier(LogisticRegression(max_iter=1000)))])

X_train, X_val, y_train, y_val = train_test_split(train_features.drop(columns=['respondent_id']), train_labels[['xyz_vaccine', 'seasonal_vaccine']], test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)

roc_auc_xyz = roc_auc_score(y_val['xyz_vaccine'], y_pred[0][:, 1])
roc_auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], y_pred[1][:, 1])
mean_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2

print(f"ROC AUC for xyz vaccine: {roc_auc_xyz}")
print(f"ROC AUC for seasonal vaccine: {roc_auc_seasonal}")
print(f"Mean ROC AUC: {mean_roc_auc}")

test_pred = model.predict_proba(test_features.drop(columns=['respondent_id']))

submission_format['xyz_vaccine'] = test_pred[0][:, 1]
submission_format['seasonal_vaccine'] = test_pred[1][:, 1]
submission_format.to_csv('/content/submission_format.csv', index=False)

