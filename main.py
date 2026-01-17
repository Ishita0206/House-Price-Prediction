import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL = "model.pkl"
PIPELINE = 'pipeline.pkl'

def buildpipeline(num_attributes,cat_attributes):

    # for numerical columns
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   #change null to median
        ("scaler", StandardScaler())   #sclae the features between 0 and 1
    ])
    # for categorical columns
    cat_pipeline = Pipeline([
        ("onehot" , OneHotEncoder(handle_unknown="ignore"))    #give 1 to present category and 0 to absent category
    ])
    #full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ('cat', cat_pipeline, cat_attributes)
    ])

    return full_pipeline


if not os.path.exists(MODEL):
    housing = pd.read_csv("housing.csv")

    housing['income_cat'] = pd.cut(housing["median_income"],
                                   bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels = [1, 2, 3, 4, 5])
    split  = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #train-test split
    for train_index, test_index in split.split(housing , housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)  

        housing_labels = housing["median_house_value"].copy()
        housing_features = housing.drop("median_house_value", axis=1)

    num_attributes = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    pipeline = buildpipeline(num_attributes, cat_attributes) 
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)
    
    joblib.dump(model, MODEL)
    joblib.dump(pipeline, PIPELINE)
    print("Model and Pipeline saved. Model is triained. congratulations!")

else:
    #let's inference the model

    model = joblib.load(MODEL)
    pipeline = joblib.load(PIPELINE)

    input_data = pd.read_csv('input.csv')
    input_prepared = pipeline.transform(input_data)
    predictions = model.predict(input_prepared) 
    input_data['median_house_value'] = predictions

    input_data.to_csv('predictions.csv', index=False)
    print("Inference done. Predictions saved to predictions.csv")

