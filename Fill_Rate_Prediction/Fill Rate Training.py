# Databricks notebook source
# MAGIC %md
# MAGIC # Job Level Model
# MAGIC This notebook loads snowflake data, cleans it, and develops a model to predict job-level show up rates (worked first shift/successful applications at job start) that will help with setting an overfill rate.  This model is at the job level and doesn't consider the attricutes of CMs that have applied.  Since it only considers successful applications at the time of job start, all cancellations prior to job start are excluded.

# COMMAND ----------

# %pip install databricks-feature-engineering
# %pip install databricks-feature-engineering==0.2.1a1
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/ml-modeling-utils
# MAGIC

# COMMAND ----------

# MAGIC %run ./_resources/mlflow-utils
# MAGIC

# COMMAND ----------

# MAGIC %run ./_resources/fill-rate-utils

# COMMAND ----------

#Imported libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import exp 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from databricks import feature_store
from databricks.feature_store import feature_table
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup

from pyspark.sql.functions import to_date, current_timestamp
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import LongType, DecimalType, FloatType
import mlflow
from mlflow.models.signature import infer_signature


# COMMAND ----------

start_date = '2023-12-01'
now = datetime.now()
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
end_date = '2024-01-01'

# sdf = jobs_query(start_date,end_date)
# display(sdf)


# COMMAND ----------

query = """
WITH tmp_calendar AS (
    SELECT DATEADD(HOUR, SEQ4(), '2023-01-01'::DATE) AS DT
    FROM TABLE (GENERATOR(ROWCOUNT => 36525))),
calendar_hours as (
    SELECT
        tc.DT                       AS CALENDAR_DATE,
        YEAR(tc.DT)                 AS CALENDAR_YEAR,
        QUARTER(tc.DT)              AS CALENDAR_QUARTER,
        MONTH(tc.DT)                AS CALENDAR_MONTH,
        WEEK(tc.DT)                 AS CALENDAR_WEEK,
        DAYOFWEEKISO(tc.DT)         AS CALENDAR_DAY_OF_WEEK,
        DAYNAME(tc.DT)              AS CALENDAR_DAY_NAME,
        DAYOFYEAR(tc.DT)            AS CALENDAR_DAY_NUMBER,
        fc.YEAR::NUMBER             AS FINANCE_YEAR,
        fc.PERIOD                   AS FINANCE_PERIOD,
        fc.WEEK_NUMBER::NUMBER      AS FINANCE_WEEK_NUMBER,
        fc.START_DATE::DATE         AS FINANCE_START_DATE,
        fc.END_DATE::DATE           AS FINANCE_END_DATE
    FROM tmp_calendar tc
    JOIN BLUECREW.DM.DM_FINANCE_CALENDAR fc
        ON tc.DT BETWEEN fc.START_DATE AND fc.END_DATE
    ORDER BY 1 ASC
),

job_dates as (
select job_id,
    job_created_at as min_date,
    job_start_date_time as max_date
    from dm.dm_jobs
    -- where shift_start >= '2018-01-01'
), 
fill_rates as (
SELECT 
    -- DATE_TRUNC('day', jsw.job_start_date_time) AS job_start,
    jsw.job_id,
    SUM(shift_needed) AS final_needed,
    SUM(shift_expected) AS final_expected,
    SUM(shift_worked) AS final_worked,
    CASE 
        WHEN SUM(shift_needed) > 0 THEN SUM(shift_expected) / SUM(shift_needed)
        ELSE NULL 
    END AS fill_rate,
    CASE 
        WHEN SUM(shift_needed) > 0 THEN SUM(shift_worked) / SUM(shift_needed)
        ELSE NULL 
    END AS work_rate,
    CASE 
        WHEN SUM(shift_expected) > 0 THEN SUM(shift_worked) / SUM(shift_expected)
        ELSE NULL 
    END AS show_up_rate
FROM 
    bluecrew.dm.fact_job_shift_worked AS jsw
WHERE 
    jsw.job_created_at >= '2024-01-01'
    AND jsw.job_start_date_time < DATEADD(HOUR, -1, SYSDATE())
    AND shift_sequence = 1
    AND jsw.job_start_date_time <= shift_start_time
    AND jsw.job_created_at <= shift_start_time
GROUP BY 
    1
ORDER BY 
    2 desc)

SELECT calendar_date, 
    j.job_id, 
    min_date,
    max_date,
    fill_rate,
    final_needed,
    datediff(HOUR, calendar_date, max_date) as lead_time_hours
FROM calendar_hours
INNER JOIN job_dates j
ON calendar_date >=min_date
AND calendar_date <= max_date
-- and calendar_date <= sysdate()
INNER JOIN fill_rates fr
ON j.job_id = fr.job_id
where final_needed > 1
order by 2, 1;
"""
sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# COMMAND ----------

display(sdf)

# COMMAND ----------

# sdf = sdf.filter((sdf.NEEDED >0)&(sdf.COMPANY_ORIGIN == 'BC')&(sdf.JOB_NEEDED_ORIGINAL_COUNT>0)&(sdf.POSTING_LEAD_TIME_DAYS>0))

# sdf = sdf.withColumn('Work', F.when(sdf.target_var == 'Worked', 1).otherwise(0))
# display(sdf)

# COMMAND ----------

sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string'))

# COMMAND ----------

# This looks up job-level features in feature store
# # For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html
fe = FeatureEngineeringClient()
model_feature_lookups = [
      #Lookup applicants
      FeatureLookup(
        table_name='feature_store.dev.job_applicant_tracker',
        lookup_key="JOB_ID",
        timestamp_lookup_key="CALENDAR_DATE",
        # feature_names=["JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"]
      ), 
      #Lookup needed
      FeatureLookup(
        table_name='feature_store.dev.job_needed_change',
        lookup_key="JOB_ID",
        timestamp_lookup_key="CALENDAR_DATE",
        # feature_names=["JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"]
      ), 
      # Calculate a new feature called `cosine_sim` - the cosine similarity between the user's work history and the current job.
      # FeatureFunction(
      #   udf_name='feature_store.dev.cosine_similarity',
      #   output_name="cosine_sim",
      #   # Bind the function parameter with input from other features or from request.
      #   # The function calculates a - b.
      #   input_bindings={"arr1":"job_schedule", "arr2": "running_schedule_array"},
      # )
]
training_set = fe.create_training_set(
    df = sdf, # joining the original Dataset, with our FeatureLookupTable
    feature_lookups=model_feature_lookups,
    exclude_columns=[], # exclude id columns as we don't want them as feature
    label="FILL_RATE"
)

training_pd = training_set.load_df()
# display(training_pd2)

# COMMAND ----------

display(training_pd)

# COMMAND ----------

df = optimize_spark(training_pd).toPandas()

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing and training preparation

# COMMAND ----------

# Defines the columns that correspond to a job and creates final dataset to split for training and testing.
cols_to_drop = ['MIN_DATE', 'MAX_DATE', 'FINAL_NEEDED' ]
# df2 = df[(df['target_var']!='Early Cancel')&(df['JOB_STATUS']!='Cancelled')]
df2 = df.copy()
df4 = df2.drop(columns=cols_to_drop)
df5 = df4.set_index('JOB_ID')
df5['current_fill_rate']= df5.apply(lambda x: x['total_applicants']/x['NEEDED'] if x['NEEDED']>0 else 1, axis = 1)
df5['current_fill_rate'].fillna(1,inplace = True)
df5 = df5[(df5['LEAD_TIME_HOURS']!=0)&(df5['total_applicants']>=0)]
df5.describe()

# COMMAND ----------

# df6 = spark.createDataFrame(df5)

# write_spark_table_to_databricks_schema(df6, 'overfill_individual_training_data', 'bluecrew.ml', mode = 'overwrite')

# COMMAND ----------

# Splits DF into train, test, and prediction dataframes
y = df5['FILL_RATE']
X = df5.drop(columns=['FILL_RATE'])

# Uses these for the training and testing
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

# Checks for missing values and determines shape of X_train
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]

print("Columns with missing values :", cols_with_missing)
print("X_train_full shape :", X_train_full.shape)
print("X_valid_full shape :", X_valid_full.shape)

# COMMAND ----------

# IDs categorical and numeric columns for use in modeling and for splitting into proper pipeline.
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 30 and X_train_full[cname].dtype in ["object", "string"]]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int32', 'int64', 'float64', 'float32','decimal']]

print('categorical columns :', categorical_cols)
print('numerical columns :', numerical_cols)

# COMMAND ----------

# IDs new columns subset
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
print(my_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Pipeline

# COMMAND ----------

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# COMMAND ----------

# DBTITLE 1,This is what we would use hyperopt to solve once we work out the shared cluster runtime issue
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from hyperopt import hp, fmin, tpe, STATUS_OK, SparkTrials
# from hyperopt.pyll.base import scope
# import shap
# import mlflow
# import numpy as np

# # Define your preprocessor here (as you did before)
# # preprocessor = ...

# # Include the model as a step in the pipeline
# my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', RandomForestRegressor())])

# def train_model(params):
#     # Update the parameter names to reflect the pipeline structure
#     params = {'model__' + key: params[key] for key in params}
    
#     # Fit the pipeline with the training data
#     my_pipeline.set_params(**params).fit(X_train, y_train)

#     # SHAP values
#     # booster = my_pipeline.named_steps['model']
#     # shap_values = shap.TreeExplainer(booster).shap_values(X_train, y=y_train)
#     # shap.summary_plot(shap_values, X_train, feature_names=display_cols, plot_size=(14,6), max_display=10, show=False)
#     # plt.savefig("summary_plot.png", bbox_inches="tight") 
#     # plt.close()
#     # mlflow.log_artifact("summary_plot.png")

#     # Predict and evaluate
#     preds = my_pipeline.predict(X_valid)
#     (rmse, mae, r2) = eval_metrics(y_valid, preds)
#     # mlflow.log_metric("rmse", rmse)
#     # mlflow.log_metric("r2", r2)
#     # mlflow.log_metric("mae", mae)
#     return {'status': STATUS_OK, 'loss': rmse}

# # Define your search space
# search_space = {
#     'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
#     'n_estimators': scope.int(hp.quniform('n_estimators', 25, 150, 25))
# }

# # mlflow.autolog()

# # with mlflow.start_run():
# # Hyperopt optimization
# spark_trials = SparkTrials(parallelism=4)
# best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=50, trials=spark_trials, rstate=np.random.default_rng(seed=42))


# COMMAND ----------

# DBTITLE 1,This will be updated once the previous code can be incorporated
# Parameters from prior hyperparameter tuning
best_params = {'max_depth': 12.0, 'n_estimators': 200.0}

# COMMAND ----------

import mlflow

mlflow.autolog()


max_depth= int(best_params['max_depth'])
n_estimators=int(best_params['n_estimators'])
# Creates initial random forest model for training
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
with mlflow.start_run():
  # Preprocessing of training data, fit model 
  my_pipeline.fit(X_train, y_train)

  # Preprocessing of validation data, get predictions
  preds = my_pipeline.predict(X_valid)


  # Evaluate the model
  #Training Performance:
  print("Training Performance:")
  (rmse, mae, r2, mape) = eval_metrics(y_train, my_pipeline.predict(X_train))

  # Print out model metrics
  print("  RMSE: %s" % rmse)
  print("  MAE: %s" % mae)
  print("  R2: %s" % r2)
  print("  MAPE: %s" % mape)

  #Test Performance:
  print("Test Performance:")
  (rmse, mae, r2, mape) = eval_metrics(y_valid, preds)

  # Print out model metrics
  print("  RMSE: %s" % rmse)
  print("  MAE: %s" % mae)
  print("  R2: %s" % r2)
  print("  MAPE: %s" % mape)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)


# COMMAND ----------

# # Define the model name for the registry
# registry_model_name = "Overfill Test"

# latest_experiment = find_latest_experiment()
# best_run_id = find_best_run_id(latest_experiment, "metrics.r2")
# model_details = register_run_as_model(registry_model_name, best_run_id)
# update_model_stage(model_details, 'Staging')

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation and Explanation

# COMMAND ----------

# Create a feature importance plot
feature_importance = FeatureImportance(my_pipeline)
feature_importance.plot(top_n_features=25)

# COMMAND ----------

# def build_prediction_intervals(y_test, preds_test, y_train, preds_train, interval: float =.7, include_residuals: bool=True):
#   """
#   Using residuals from the model predictions, determines the prediction interval that would capture a given percent of all correct predictions

#   Parameters:
#   - y_test: actual test target variable values, should have a primary key as the index to join with X_test
#   - preds_test: predicted test target variable values
#   - y_train: actual training target variable values, should have a primary key as the index to join with X_train
#   - preds_train: predicted training target variable values
#   - interval: percent of residuals to capture
#   - include_residuals: whether or not to include columns with the residuals for each prediction

#   Returns: 
#   prediction_interval_df: pandas df with the index of y_test, y_train, the upper and lower predictions for each, and a column denoting which dataset it came from
#   """
  


#   return prediction_interval_df

# COMMAND ----------

# Builds DataFrame with predicted and actual values for test set
# Determines the range of values to consider for upper and lower bounds of a prediction interval.  Want this to be large enough to cover the actual work rate most of the time, but remain small enough to be useful
interval = .7
a = y_valid.reset_index()
a = a.rename(columns={'Work':'Actual_Show_Up_Rate'})
a['Predicted_Show_Up_Rate'] = preds
a['Delta'] = abs(a['Predicted_Show_Up_Rate'] - a['Actual_Show_Up_Rate'])
a['Signed_Delta'] = a['Predicted_Show_Up_Rate'] - a['Actual_Show_Up_Rate']

# 50% interval
a['lowq'] = a['Signed_Delta'].quantile((1-interval)/2)
a['highq'] = a['Signed_Delta'].quantile(1-(1-interval)/2)

a['Dataset']="Test"
print(a)

# COMMAND ----------

# Builds DataFrame with predicted and actual values for training set
b = y_train.reset_index()
b = b.rename(columns={'Work':'Actual_Show_Up_Rate'})
b['Predicted_Show_Up_Rate'] = my_pipeline.predict(X_train)
b['Delta'] = abs(b['Predicted_Show_Up_Rate'] - b['Actual_Show_Up_Rate'])
b['Signed_Delta'] = b['Predicted_Show_Up_Rate'] - b['Actual_Show_Up_Rate']
b['lowq'] = b['Signed_Delta'].quantile((1-interval)/2)
b['highq'] = b['Signed_Delta'].quantile(1-(1-interval)/2)
b['Dataset']="Training"
print(b)

# COMMAND ----------

# Merges job data to look at characteristics associated with predictions
c = pd.concat([a,b])

# creates the upper and lower bounds of the prediction interval
c['lower_guess']=(c['Predicted_Show_Up_Rate'] + c['lowq']).clip(lower = .01) #can't have negative numbers
c['upper_guess']=(c['Predicted_Show_Up_Rate'] + c['highq']).clip(upper = 1) #can't have numbers over 1

#merges with df5 to incorporate job data
eda_df = c.merge(df5, left_on='JOB_ID', right_index = True, how = 'left').sort_values('JOB_NEEDED_ORIGINAL_COUNT', ascending = True)
eda_df2 = eda_df[eda_df['Actual_Show_Up_Rate']==0]



# COMMAND ----------

overfill_added = eda_df.merge(df[['JOB_OVERFILL', 'TOTAL_SUCCESSFUL_SIGN_UPS', 'JOB_ID']], on='JOB_ID', how = 'left')
overfill_added['JOB_OVERFILL'].fillna(0,inplace=True)
overfill_added['PAST_PREDICTED_Show_Up_Rate']=overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']+overfill_added['JOB_OVERFILL'])
# original needed / original needed + overfill
overfill_added_test = overfill_added[overfill_added['Dataset']=='Test']
print("Past performance considering all jobs in test data:")

(rmse, mae, r2, mape) = eval_metrics(overfill_added_test.loc[overfill_added_test['START_TIME_LOCAL']<=end_date,'Actual_Show_Up_Rate'], overfill_added_test.loc[overfill_added_test['START_TIME_LOCAL']<=end_date,'PAST_PREDICTED_Show_Up_Rate'])

# Print out model metrics
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

overfill_added2 = overfill_added_test[overfill_added_test['JOB_OVERFILL']!=0]

print("Past performance on jobs with overfill added:")
(rmse, mae, r2, mape) = eval_metrics(overfill_added2['Actual_Show_Up_Rate'], overfill_added2['PAST_PREDICTED_Show_Up_Rate'])

# Print out model metrics
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)



# COMMAND ----------

#calculate costs associated with each prediction
overfill_added['Overfill_to_100']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['Actual_Show_Up_Rate']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_to_100_perc'] = 1/overfill_added['Actual_Show_Up_Rate'] - 1
overfill_added['Overfill_Recommendation']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['Predicted_Show_Up_Rate']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_rec_perc'] = 1/overfill_added['Predicted_Show_Up_Rate'] - 1
overfill_added['Overfill_Rec_Upper_Bound']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['lower_guess']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_upper_perc'] = 1/overfill_added['lower_guess'] - 1
overfill_added['Overfill_Rec_Lower_Bound']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['upper_guess']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_lower_perc'] = 1/overfill_added['upper_guess'] - 1

overfill_added['Overfill_Rec_to_Ideal']=overfill_added['Overfill_Recommendation']-overfill_added['Overfill_to_100']
overfill_added['Overfill_Past_to_Ideal']=overfill_added['JOB_OVERFILL']-overfill_added['Overfill_to_100']

overfill_added['Overfill_Prediction_Lost_Profit'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']*-25 if row['Overfill_Rec_to_Ideal']<0 else 0, axis=1).apply(np.round)
overfill_added['Overfill_Prediction_CM_Payout'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Rec_to_Ideal']>=0 else 0, axis=1).apply(np.round)

overfill_added['Overfill_Past_Rec_Lost_Profit'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']*-25 if row['Overfill_Past_to_Ideal']<0 else 0, axis=1).apply(np.round)
overfill_added['Overfill_Past_Rec_CM_Payout'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Past_to_Ideal']>=0 else 0, axis=1).apply(np.round)

overfill_added['Overfill_Prediction_Cost'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Rec_to_Ideal']>=0 else row['Overfill_Rec_to_Ideal']*-25, axis=1).apply(np.round)
overfill_added['Overfill_Past_Rec_Cost'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Past_to_Ideal']>=0 else row['Overfill_Past_to_Ideal']*-25, axis=1).apply(np.round)
overfill_added[overfill_added['Actual_Show_Up_Rate']>0]
overfill_added[(overfill_added['Dataset']=='Future')&(overfill_added['JOB_ID']==329636)]

# COMMAND ----------



# COMMAND ----------

cost_estimate = overfill_added[overfill_added['Actual_Show_Up_Rate']>0]
print(cost_estimate.shape)
print('Model Overfill Prediction_Cost:' + str(cost_estimate['Overfill_Prediction_Cost'].sum()))
print('Past Overfill Prediction_Cost:' + str(cost_estimate['Overfill_Past_Rec_Cost'].sum()))
print('Model Overfill Lost Profit:' + str(cost_estimate['Overfill_Prediction_Lost_Profit'].sum()))
print('Past Overfill Lost Profit:' + str(cost_estimate['Overfill_Past_Rec_Lost_Profit'].sum()))
print('Model Overfill CM Payout:' + str(cost_estimate['Overfill_Prediction_CM_Payout'].sum()))
print('Past Overfill CM Payout:' + str(cost_estimate['Overfill_Past_Rec_CM_Payout'].sum()))

# COMMAND ----------

x_array = np.arange(0,1,.01)
x_array2 = np.arange(0,3500,1)

# COMMAND ----------

#Used to see if job size impacted model accuracy.  Used this to determine jobs with 1 or 2 people were throwing off the model.
sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Overfill_Prediction_Cost",
                    y="Overfill_Past_Rec_Cost",
                    hue = 'JOB_NEEDED_ORIGINAL_COUNT',
                    data=overfill_added[(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']<=50)&(overfill_added['Dataset']!="Future")])
plt.axis('equal')
# plt.xlim(0,14000)
plt.title("Overfill Costs for Jobs with Less Than 50 Requested")
plt.plot(x_array2, x_array2, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.show()

# COMMAND ----------

#Used to see if job size impacted model accuracy.  Used this to determine jobs with 1 or 2 people were throwing off the model.
sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Overfill_Prediction_Lost_Profit",
                    y="Overfill_Past_Rec_Lost_Profit",
                    hue = 'JOB_NEEDED_ORIGINAL_COUNT',
                    data=overfill_added[overfill_added['JOB_NEEDED_ORIGINAL_COUNT']<=50])
plt.axis('equal')
# plt.xlim(0,14000)
plt.title("Overfill Lost Profit for Jobs with Less Than 50 Requested")
plt.plot(x_array2, x_array2, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.show()

# COMMAND ----------

#Used to see if job size impacted model accuracy.  Used this to determine jobs with 1 or 2 people were throwing off the model.
sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Overfill_Prediction_CM_Payout",
                    y="Overfill_Past_Rec_CM_Payout",
                    hue = 'JOB_NEEDED_ORIGINAL_COUNT',
                    data=overfill_added[overfill_added['JOB_NEEDED_ORIGINAL_COUNT']<=50])
plt.axis('equal')
# plt.xlim(0,14000)
plt.title("Overfill CM Payouts for Jobs with Less Than 50 Requested")
plt.plot(x_array2, x_array2, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.show()

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Actual_Show_Up_Rate",
                    y="Predicted_Show_Up_Rate",
                    hue = 'Delta',
                    data=overfill_added_test)
plt.plot(x_array, x_array, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.xlabel('Actual Show Up Rate')
plt.ylabel('Predicted Show Up Rate')
plt.show()

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Actual_Show_Up_Rate",
                    y="PAST_PREDICTED_Show_Up_Rate",
                    hue = 'Dataset',
                    data=overfill_added2)
plt.show()

# COMMAND ----------

overfill_added['as_of_date']=datetime.now()
df = spark.createDataFrame(overfill_added)
df.createOrReplaceTempView('data')


# COMMAND ----------

df.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL_TRAINING').save()


# COMMAND ----------


write_spark_table_to_databricks_schema(df, 'overfill_training_data', 'bluecrew.ml', mode = 'append')

# COMMAND ----------

display(df[df['Actual_Show_Up_Rate']==1])

# COMMAND ----------

iterations = 10
a = X_valid.copy()
X_valid2 = X_valid.copy()
a['Predicted_Show_Up_Rate'] = my_pipeline.predict(X_valid2)
for i in range(0,iterations):
  X_valid2['JOB_WAGE']=X_valid2['JOB_WAGE']+1
  preds = my_pipeline.predict(X_valid2)
  a['Prediction_'+str(i)] = preds

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))

sns.scatterplot(x="Predicted_Show_Up_Rate",
                    y=f"Prediction_9",
                    data=a)
plt.plot(x_array, x_array, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.xlabel('Original Predicted Show Up Rate')
plt.ylabel('New Predicted Show Up Rate')
plt.show()

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
for i in range(0,iterations):
  sns.scatterplot(x="Predicted_Show_Up_Rate",
                      y=f"Prediction_{i}",
                      size = int(i),
                      hue = int(i),
                      data=a)
plt.plot(x_array, x_array, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.xlabel('Original Predicted Show Up Rate')
plt.ylabel('New Predicted Show Up Rate')
plt.show()

# COMMAND ----------

display(df[df['Dataset']=='Future'])

# COMMAND ----------


