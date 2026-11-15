"""Rebuild superkart_model.joblib exactly as the notebook does (tuned Random Forest, full data)."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REFERENCE_YEAR = 2025
df = pd.read_csv("SuperKart.csv")
df["Product_Id_char"] = df["Product_Id"].str[:2]
df["Store_Age_Years"] = REFERENCE_YEAR - df["Store_Establishment_Year"]
df["Product_Type_Category"] = df["Product_Id_char"].map(
    {"FD": "Perishables", "DR": "Non Perishables", "NC": "Non Perishables"})
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

feature_cols = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
                "Product_MRP", "Store_Size", "Store_Location_City_Type", "Store_Type",
                "Product_Id_char", "Store_Age_Years", "Product_Type_Category"]
categorical = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
               "Store_Type", "Product_Id_char", "Product_Type_Category"]

model = Pipeline([
    ("preprocessor", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
        remainder="passthrough")),
    ("model", RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5,
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)),
])
model.fit(df[feature_cols], df["Product_Store_Sales_Total"])
joblib.dump(model, "superkart_model.joblib")
print("saved superkart_model.joblib")
