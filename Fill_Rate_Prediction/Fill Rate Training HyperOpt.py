# Databricks notebook source
# MAGIC %md
# MAGIC # Individual Level Model Description
# MAGIC This notebook runs util notebooks, queries the foreign catalog to obtain individual level application event data, cleans it, and develops a model to predict individual-level show up rates.  It answers the question, "if someone applied, what was the outcome of that application?" by first checking to see if they worked the first shift of the applied to job.  If not, it checks when their latest cancellation occured and categorizes them into early cancellation, SNC, and NCNS.  This model will supplement the job-level overfill prediction model and apply its predictions to the applicant population.

# COMMAND ----------

# MAGIC %md
# MAGIC # Library installs and util runs

# COMMAND ----------

# %pip install databricks-feature-engineering
# %pip install databricks-feature-engineering==0.2.1a1
%pip install explainerdashboard
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/ml-modeling-utils

# COMMAND ----------

# MAGIC
# MAGIC %run ./_resources/mlflow-utils

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



from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql import SparkSession
from math import exp
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score
# from imblearn.under_sampling import RandomUnderSampler

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope

import mlflow
from xgboost import XGBClassifier, XGBRegressor

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking import MlflowClient


# COMMAND ----------

start_date = '2023-12-01'
now = datetime.now()
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
end_date = '2024-01-01'

# COMMAND ----------

query = """
WITH tmp_calendar AS (
        SELECT DATEADD(MINUTE, SEQ4(), '2023-01-01'::DATE) AS DT
        FROM TABLE (GENERATOR(ROWCOUNT => 36525*60))),
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
        and jsw.job_id not in (select job_id from dm.fact_job where company_origin = 'EB')
    GROUP BY 
        1
    ORDER BY 
        2 desc)
-- select * from calendar_hours
SELECT calendar_date, 
    j.job_id, 
    min_date,
    max_date,
    fill_rate,
    final_needed,
    datediff(MINUTE, calendar_date, max_date)/60 as lead_time_hours
FROM calendar_hours
INNER JOIN job_dates j
ON calendar_date >=date_trunc(MINUTE,min_date)
AND calendar_date <= max_date
and (calendar_date = dateadd(MINUTE,1, date_trunc(MINUTE,min_date)) or MINUTE(calendar_date) = 0 or calendar_date = min_date)
-- and calendar_date <= sysdate()
INNER JOIN fill_rates fr
ON j.job_id = fr.job_id
where final_needed > 1 
order by 2, 1;
"""
sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string'))

display(sdf)

# COMMAND ----------



# COMMAND ----------

# This looks up job-level features in feature store
# # For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html
fe = FeatureEngineeringClient()
model_feature_lookups = [
      FeatureLookup(
      table_name='feature_store.dev.jobs_data',
        # feature_names=['COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'WAGE_DELTA', 'SCHEDULE_NAME_UPDATED',"ELIGIBLE_USERS", "ACTIVE_USERS_7_DAYS", "COUNTY", "ELIGIBLE_CMS_1_MILE", "ELIGIBLE_CMS_5_MILE","ELIGIBLE_CMS_10_MILE", 'ELIGIBLE_CMS_15_MILE', "ACTIVE_CMS_1_MILE", "ACTIVE_CMS_5_MILE", "ACTIVE_CMS_10_MILE", "ACTIVE_CMS_15_MILE", "JOB_TYPE_TITLE_COUNT", "TOTAL_JOB_COUNT", "TOTAL_CMS_REQUIRED", "CM_COUNT_RATIO"],
        lookup_key="JOB_ID",
        timestamp_lookup_key="CALENDAR_DATE"),
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
      FeatureLookup(
        table_name='feature_store.dev.job_views_and_applications',
        lookup_key="JOB_ID",
        timestamp_lookup_key="CALENDAR_DATE",
        # feature_names=["JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"]
      ),
      #Lookup overfill
      FeatureLookup(
        table_name='feature_store.dev.job_overfill_change',
        lookup_key="JOB_ID",
        timestamp_lookup_key="CALENDAR_DATE",
        # feature_names=["JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"]
      ), 
      # Calculate a new feature called `cosine_sim` - the cosine similarity between the user's work history and the current job.
      # FeatureFunction(
      #   udf_name='feature_store.dev.double_division',
      #   output_name="current_fill_rate",
      #   # Bind the function parameter with input from other features or from request.
      #   # The function calculates a - b.
      #   input_bindings={"numerator":"total_applicants", "denominator": "NEEDED"},
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
# display(training_pd)

# COMMAND ----------

training_pd.createOrReplaceTempView("training_data")

# COMMAND ----------

window_size = 4
training_df = spark.sql(f"""
              with temp as (  
                select *, 
                  case when needed > 0 then total_applicants/needed else 1 end as current_fill_rate,
                  needed+overfill as needed_and_overfill,
                  case when BLUESHIFT_REQUEST_ID is not null then 1 else 0 end as blueshift,
                  case when FORKED_FROM is not null then 1 else 0 end as forked,
                  case when needed_and_overfill > 0 then total_applicants/needed_and_overfill else 1 end as overfill_rate,
                  case when POSTING_LEAD_TIME_DAYS <= 3 then '0-3'
                    when POSTING_LEAD_TIME_DAYS <= 7 then '4-7'
                    when POSTING_LEAD_TIME_DAYS <= 14 then '7-14'
                    when POSTING_LEAD_TIME_DAYS <= 30 then '15-30'
                    else '31+' end as POSTING_LEAD_TIME_BINS,
                  case when lead_time_hours < 48 then -1
                    when requested > 0 then confirmed/requested
                    else 0 end as current_confirmed_rate

                  
                from training_data
              )
              select *,
                avg(total_applicants) over(partition by job_id order by calendar_date rows between {window_size}-1 preceding and current row) as avg_applicants_last_{window_size},
                avg(NEEDED) over(partition by job_id order by calendar_date rows between {window_size}-1 preceding and current row) as avg_needed_last_{window_size},
                avg(current_fill_rate) over(partition by job_id order by calendar_date rows between {window_size}-1 preceding and current row) as avg_fill_rate_last_{window_size},
                max(lead_time_hours) over(partition by job_id) as total_lead_time,
                lag(current_fill_rate) over(partition by job_id order by calendar_date) as prior_hour_fill_rate,
                avg(lag(case when overfill_rate < 1 then total_applicants - lag(total_applicants) over(partition by job_id order by calendar_date) else null end) over(partition by job_id order by calendar_date)) over(partition by job_id order by calendar_date) as fill_rate_when_unfilled,
                lag(total_applicants) over(partition by job_id order by calendar_date) as prior_hour_applicants,
                first_value(lead(total_applicants) over(partition by job_id order by calendar_date)) over(partition by job_id order by calendar_date) as first_hour_applicants,
                first_value(lead(current_fill_rate) over(partition by job_id order by calendar_date)) over(partition by job_id order by calendar_date) as first_hour_fill_rate,
                case when views > 0 then apps/views else 0 end as views_per_app
              from temp
                        """)
                        
# display(training_df)

# df2['normalized_lead_time']=df2.apply(lambda x: 1 - x['LEAD_TIME_HOURS']/(x['total_lead_time'] if x['total_lead_time']>0 else 1),axis = 1)
# df2['needed_and_overfill'] = df2['NEEDED']+df2['OVERFILL']
# df2['blueshift'] = df2['BLUESHIFT_REQUEST_ID'].apply(lambda x: 1 if x==x else 0)
# df2['forked'] = df2['FORKED_FROM'].apply(lambda x: 1 if x==x else 0)
# df2['overfill_rate']=df2.apply(lambda x: x['total_applicants']/(x['NEEDED']+x['OVERFILL']) if (x['NEEDED']+x['OVERFILL'])>0 else 0, axis = 1)
# df2['POSTING_LEAD_TIME_BINS']= df2['POSTING_LEAD_TIME_DAYS'].apply(lambda x: "0-3" if x <=3 else "4-7" if x<=7 else "7-14" if x<=14 else "15-30" if x<=30 else "31+")
# df4 = df2.drop(columns=cols_to_drop)
# df5 = df4.set_index('JOB_ID')
# df5['current_fill_rate']= df5.apply(lambda x: x['total_applicants']/x['NEEDED'] if x['NEEDED']>0 else -1, axis = 1)
# df5['current_confirmed_rate']= df5.apply(lambda x: x['confirmed']/x['total_applicants'] if x['total_applicants']>0 else 0 if x['requested']>0 else -1, axis = 1)
# df5['prior_1_fill_rate'] = df5.groupby(['JOB_ID'])['current_fill_rate'].shift(1)
# df5['applicants_prior_hour'] = df5.groupby(['JOB_ID'])['total_applicants'].shift(1)
# df5['applicant_changes_last_hour'] = df['total_applicants']-df['applicants_prior_hour']
# df5['avg_applicants_last_4_hours'] = df5.groupby('JOB_ID')['total_applicants'].rolling(4).mean().reset_index(0,drop=True)
# df5['avg_fill_rate_last_4_hours'] = df5.groupby('JOB_ID')['current_fill_rate'].rolling(4).mean().reset_index(0,drop=True)

# COMMAND ----------


df = optimize_spark(training_df).toPandas()

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))


# COMMAND ----------

# Defines the columns that correspond to a job and creates final dataset to split for training and testing.
cols_to_drop = ['MIN_DATE', 'MAX_DATE', 'FINAL_NEEDED', 'JOB_OVERFILL', 'JOB_TEMPLATE_ID', 'EXTERNAL_JOB_ID', 'JOB_TEMPLATE_ID', 'JOB_TEMPLATES_EXTERNAL_ID', 'BLUESHIFT_REQUEST_ID', 'FORKED_FROM', 'JOB_POSTER_ID', 'JOB_POSTER_FULL_NAME', 'JOB_POSTER_EMAIL', 'JOB_TYPE_ID', 'JOB_DESCRIPTION', 'SCHEDULE_ID', 'SCHEDULE_NAME', 'JOB_STATUS_ENUM', 'NOTES_JOBS_DATA', 'JOB_BATCH_SIZE', 'JOB_REASON_CODE', 'JOB_REASON_TEXT', 'JOB_ADDRESS', 'JOB_ADDRESS_LINE_TWO', 'JOB_CITY', 'COUNTY','JOB_TITLE', 'JOB_ZIPCODE', 'JOB_ADDRESS_LATITUDE', 'JOB_ADDRESS_LONGITUDE', 'JOB_REGION_ID', 'ELIGIBLE_CMS_10_MILE', 'ELIGIBLE_CMS_15_MILE', 'ACTIVE_CMS_10_MILE', 'ACTIVE_CMS_15_MILE', 'COMPANY_ID', 'POSITION_ID']


df2 = df.copy()

df4 = df2.drop(columns=cols_to_drop)
df5 = df4.set_index('JOB_ID')
df5['fill_rate_when_unfilled_2'] = df5.apply(lambda x: x['fill_rate_when_unfilled']/x['NEEDED'] if x['NEEDED']>0 else 0, axis = 1)
df5 = df5[(df5['LEAD_TIME_HOURS']<=(24*7))&(df5['FILL_RATE']< 2.5)&(df5['POSTING_LEAD_TIME_DAYS']>0)&(df5['JOB_STATUS']!='Cancelled')]
df5.describe()

# COMMAND ----------

df5['fill_rate_when_unfilled_2'].max()

# COMMAND ----------

# df6 = df5[((df5['current_fill_rate'] - df5['prior_1_fill_rate'] > .8)|(df5['current_fill_rate'] - df5['prior_1_fill_rate'] < -.8))&(df5['LEAD_TIME_HOURS']==0)]
# # df6 = df5.copy()
# # df6[['current_fill_rate', 'FILL_RATE', 'NEEDED', 'total_applicants']]
# # df6['lagged_values'] = df6.groupby(['JOB_ID'])['current_fill_rate'].shift(1)
# print(list(df6.index))
# df6

# COMMAND ----------

# df7 = df5[df5['FILL_RATE']>=3]
# print(set(df7.index))

# COMMAND ----------

# df5[df5.index == '345827']
# df_test = spark.sql(""" 
#                     select * from feature_store.dev.job_applicant_tracker
#                     where job_id = 327529
#                     """)
# display(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training

# COMMAND ----------

mlflow.autolog(silent=True)
mlflow.set_registry_uri('databricks-uc')
mlflow.set_tracking_uri("databricks")
# mlflow.set_experiment("antwoineflowers/databricks-uc/jobmatchscoring")
model_name = 'fill_rate_test'

 # Separate features and target variable
X = df5.drop('FILL_RATE', axis=1)
y = df5['FILL_RATE']

# Create Preprocessing Pipeline

# Identify numerical and categorical columns (excluding the label column)
numerical_cols = df5.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.drop('FILL_RATE')

categorical_cols = df5.select_dtypes(include=['object', 'category']).columns

# Preprocessing for numerical data: impute nulls with 0 and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute with the most frequent and apply one-hot encoding
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

# from imblearn.under_sampling import RandomUnderSampler
# under_sampler = RandomUnderSampler(sampling_strategy='not minority')
# X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

# Split data into training, validation, and test sets
job_ids = np.array(list(set(df5.index)))

rem_job_ids = np.random.choice(job_ids,int(len(job_ids)*.6), replace=False)
test_job_ids = np.random.choice(rem_job_ids,int(len(job_ids)*.5), replace=False)

X_train =X.loc[~X.index.isin(rem_job_ids)]
X_val = X.loc[(X.index.isin(rem_job_ids))&(~X.index.isin(test_job_ids))]
X_test =X.loc[X.index.isin(test_job_ids)]

y_train =y.loc[~y.index.isin(rem_job_ids)]
y_val = y.loc[(y.index.isin(rem_job_ids))&(~y.index.isin(test_job_ids))]
y_test =y.loc[y.index.isin(test_job_ids)]
# X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=9986)
# X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=72688)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the hyperparameter space
space = {
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'subsample': hp.uniform('subsample', 0.7, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    # 'objective': 'binary:logistic',
    # 'eval_metric': 'r2',
}

# Objective function for hyperparameter optimization
def objective(params):
    with mlflow.start_run(run_name='overfill_training', nested=True): 
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Initialize XGBRegressor
        model = XGBRegressor(**params 
        )

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Predict and evaluate using the probabilities of the positive class
        predictions = pipeline.predict(X_val)

        print("Training Performance:")
        (rmse, mae, r2, mape) = eval_metrics(y_train, pipeline.predict(X_train))

        # Print out model metrics
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  MAPE: %s" % mape)

        #Test Performance:
        print("Test Performance:")
        (rmse, mae, r2, mape) = eval_metrics(y_val, predictions)

        # Print out model metrics
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  MAPE: %s" % mape)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        return {'loss': -r2, 'status': STATUS_OK, 'model': pipeline, 'rmse': rmse, 'MAE': mae, 'mape':mape}

# Hyperparameter optimization with Hyperopt
with mlflow.start_run(run_name='Hyperparameter Optimization') as parent_run:
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,  
        algo=tpe.suggest,
        max_evals=5,  
        trials=trials
    )

# Fetch the details of the best run
best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
best_params = space_eval(space, best)

# Print out the best parameters and corresponding loss
print(f"Best parameters: {best_params}")
print(f"Best eval auc: {best_run['loss']}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Recreating and Logging the model
# MAGIC ##### Once the model has been through the hyperopt pipeline, I rerun the model with the best parameters to easily find the model for logging

# COMMAND ----------

mlflow.sklearn.autolog()
# Log the best hyperparameters
with mlflow.start_run(run_name='best_model_run'):
  model = objective(best_params)
  mlflow.set_tag("mlflow.best_model_run", "best_run")
final_model = model['model']  
r2 = -1*model['loss']



# COMMAND ----------

# # Define the model name for the registry
registry_model_name = "bluecrew.ml.fill_rate_test"

client = MlflowClient()
client.delete_registered_model(name=registry_model_name)

latest_experiment = find_latest_experiment()
best_run_id = find_best_run_id(latest_experiment, "metrics.r2")
# Uncomment this to register the model
model_details = register_run_as_model(registry_model_name, best_run_id)
# update_model_stage(model_details, 'Staging')

# COMMAND ----------

# MAGIC %md
# MAGIC # Sample code to show model loading

# COMMAND ----------

#this doesn't come with predict_proba.  I think you could build a custom model wrapper to fix this, but I didn't.
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{registry_model_name}/1")
#this allows you to get the predict_proba method
model2 = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name}/1")


# pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{registry_model_name}/1")

# COMMAND ----------

# Check model Signature
# import mlflow
# model_uri = client.get_model_version_download_uri('toy-model','10')
model_info = mlflow.models.get_model_info(f"models:/{registry_model_name}/1")
sig_dict = model_info._signature_dict
sig_dict['inputs']


# COMMAND ----------

import ast
true = True
false = False
schema_dict = {}
input_dict_list = eval(sig_dict['inputs'])
# print(input_dict)
for value in input_dict_list:
  # print(value)
  schema_dict[value['name']]=value['type']
schema_dict

# COMMAND ----------

# col_list = [key for key in schema_dict]
# print(col_list)
df_test = df5.copy()

for key in schema_dict:
  if df_test.dtypes[key]==schema_dict[key]:
    # print('ok')
    pass
  else:
    print(f"df_type: {df_test.dtypes[key]}, schema_req: {schema_dict[key]}")
    # df_test[key].astype(schema_dict[key])

# df_test['predicted_prob'] = model2.predict_proba(df_test[[key for key in schema_dict]])[:,1]
df_test['prediction'] = model2.predict(df_test[[key for key in schema_dict]])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction analysis

# COMMAND ----------

df_test[['prediction', 'FILL_RATE']]

# COMMAND ----------

predictions = pd.Series(model2.predict(X_val))

# COMMAND ----------

X = df5.drop('FILL_RATE', axis=1)
y = df5['FILL_RATE']
display(X)

# COMMAND ----------

predictions = pd.Series(final_model.predict(X_test))
# predictions = pd.Series(final_model.predict(X_val))
display(predictions)

# COMMAND ----------

# y2 = df5['Work'].reset_index()
# Creating a dictionary by passing Series objects as values
# y2['Predictions']=predictions
x2 = pd.concat([X_test.reset_index(),predictions,y_test.reset_index()['FILL_RATE']],axis = 1)
x2.rename(columns={0:'prediction'}, inplace = True)
# y3 = y2.merge(X.reset_index(), on = 'Unique_ID',how = 'inner')
display(x2)

# x2.groupby(['Work']).count()
 

# COMMAND ----------

sns.kdeplot(
   data=x2, x='prediction', 
   fill=True, common_norm=False,
   alpha=.5, linewidth=0,
)
sns.kdeplot(
   data=x2, x='FILL_RATE', 
   fill=True, common_norm=False,
   alpha=.5, linewidth=0,
)
plt.show()
print("Orange is actual.  Blue is prediction")

# COMMAND ----------

# x2['ever_worked'].fillna(0,inplace = True)
sns.kdeplot(
   data=x2, x='prediction', 
   # hue="ever_worked",
   fill=True, common_norm=False,
   alpha=.5, linewidth=0,
)

# COMMAND ----------

# sns.histplot(data=x2[x2['CLUSTER_FINAL']!= '-1'], x='predicted_probability', hue="CLUSTER_FINAL",palette="Spectral",
#    alpha=.5, linewidth=0,bins=10, multiple='stack')
# plt.show()

# COMMAND ----------

sns.histplot(data=x2, x='prediction',
   alpha=.5, linewidth=0,bins=10, multiple='stack')
plt.show()

# COMMAND ----------

# jobs_and_users = x2.merge(df3, on = 'Unique_ID', how = 'left')
# job_modeling = jobs_and_users[['JOB_ID','predicted_probability','Work']].groupby('JOB_ID').agg({'Work': 'sum', 'JOB_ID':'count', 'predicted_probability': 'sum'})
# job_modeling['work_rate']= job_modeling['Work']/job_modeling['JOB_ID']
# job_modeling['predicted_work_rate']= job_modeling['predicted_probability']/job_modeling['JOB_ID']
# job_modeling2 = job_modeling[job_modeling['JOB_ID']>3]
# display(job_modeling)

# COMMAND ----------

(rmse, mae, r2, mape) = eval_metrics(x2['FILL_RATE'], x2['prediction'])

# Print out model metrics
print("Comparing expected fill rate:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)

# (rmse, mae, r2, mape) = eval_metrics(job_modeling['work_rate'], job_modeling['predicted_work_rate'])

# # Print out model metrics
# print("Comparing Work Rate:")
# print("  RMSE: %s" % rmse)
# print("  MAE: %s" % mae)
# print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)

(rmse, mae, r2, mape) = eval_metrics(x2.loc[x2['LEAD_TIME_HOURS']>48,'FILL_RATE'], x2.loc[x2['LEAD_TIME_HOURS']>48,'prediction'])

# Print out model metrics
print("Comparing Fill Rate for jobs before confirmations requests:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)

(rmse, mae, r2, mape) = eval_metrics(x2.loc[x2['LEAD_TIME_HOURS']==x2['total_lead_time'],'FILL_RATE'], x2.loc[x2['LEAD_TIME_HOURS']==x2['total_lead_time'],'prediction'])

# Print out model metrics
print("Comparing Fill Rate for jobs at creation:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)


(rmse, mae, r2, mape) =  eval_metrics(x2['FILL_RATE'], x2['current_fill_rate'])

# Print out model metrics
print("Comparing current fill rate:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)

lead_time = 0
(rmse, mae, r2, mape) = eval_metrics(x2.loc[x2['LEAD_TIME_HOURS']==lead_time,'FILL_RATE'], x2.loc[x2['LEAD_TIME_HOURS']==lead_time,'prediction'])

# Print out model metrics
print(f"Comparing Predicted Fill Rate for jobs {lead_time} hours before job start:")
# print(x2.loc[x2['LEAD_TIME_HOURS']==lead_time,'FILL_RATE'].shape)
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)

(rmse, mae, r2, mape) = eval_metrics(x2.loc[x2['LEAD_TIME_HOURS']==lead_time,'FILL_RATE'], x2.loc[x2['LEAD_TIME_HOURS']==lead_time,'current_fill_rate'])

# Print out model metrics
print(f"Comparing current fill rate for jobs {lead_time} hours before job start:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)

# COMMAND ----------

sns.scatterplot(data=x2, x='current_fill_rate',y='prediction',
   alpha=.1)
plt.show()

# COMMAND ----------

sns.scatterplot(data=x2, x='FILL_RATE',y='prediction', hue = "POSTING_LEAD_TIME_DAYS",
   alpha=.1)
plt.show()

# COMMAND ----------

x3 = x2[x2['LEAD_TIME_HOURS']==1]

# COMMAND ----------

sns.scatterplot(data=x3, x='POSTING_LEAD_TIME_DAYS',y='FILL_RATE', hue = "prediction",
   alpha=.1)
plt.show()

# COMMAND ----------

print(list(x3.columns))

# COMMAND ----------

# sns.boxenplot(data=x3, y='POSTING_LEAD_TIME_BINS',x='FILL_RATE')
# plt.show()
# sns.boxenplot(data=x3, y='JOB_REGION',x='FILL_RATE')
# plt.show()
plt.figure(figsize=(10,10))
sns.violinplot(data=x3, y='POSTING_LEAD_TIME_BINS',x='FILL_RATE',split=True, inner="quart")
plt.show()
plt.figure(figsize=(10,10))
sns.violinplot(data=x3, y='SCHEDULE_NAME_UPDATED',x='FILL_RATE', split=True, inner="quart")
plt.show()

# COMMAND ----------

cols_to_plot = ['CALENDAR_DATE', 'COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'JOB_TYPE_TITLE_COUNT', 'TOTAL_JOB_COUNT', 'TOTAL_CMS_REQUIRED', 'WAGE_DELTA', 'FILL_RATE']

cols_to_plot2 = ['WAGE_DELTA_PERCENT', 'JOB_TYPE', 'JOB_WAGE', 'JOB_NEEDED_LAST_COUNT', 'JOB_SHIFTS','FILL_RATE' ]

cols_to_plot3 = ['INVITED_WORKER_COUNT', 'JOB_IS_APPLICATION', 'SCHEDULE_NAME_UPDATED', 'POSTING_LEAD_TIME_DAYS', 'ELIGIBLE_USERS', 'ACTIVE_USERS_7_DAYS', 'FILL_RATE']

cols_to_plot4 = ['ELIGIBLE_CMS_1_MILE', 'ACTIVE_CMS_1_MILE',  'CM_COUNT_RATIO', 'total_applicants', 'NEEDED', 'OVERFILL', 'needed_and_overfill','FILL_RATE']


# sns.pairplot(x3[cols_to_plot])
# plt.show()

# sns.pairplot(x3[cols_to_plot2])
# plt.show()


# sns.pairplot(x3[cols_to_plot3])
# plt.show()

# sns.pairplot(x3[cols_to_plot4])
# plt.show()

# COMMAND ----------

x3['FILL_RATE'].dtype

# COMMAND ----------

sns.heatmap(x3[cols_to_plot].corr(numeric_only = True))
plt.show()
sns.heatmap(x3[cols_to_plot2].corr(numeric_only = True))
plt.show()
sns.heatmap(x3[cols_to_plot3].corr(numeric_only = True))
plt.show()
sns.heatmap(x3[cols_to_plot4].corr(numeric_only = True))
plt.show()

# COMMAND ----------

sns.scatterplot(data=x2, x='current_fill_rate',y='FILL_RATE', hue = 'LEAD_TIME_HOURS',
   alpha=.1)
plt.show()

# COMMAND ----------

sns.scatterplot(data=x2, x='NEEDED',y='FILL_RATE',
   alpha=.1)
plt.show()

# COMMAND ----------

x2['Delta'] = x2['FILL_RATE']-x2['prediction']
sns.scatterplot(data=x2, x='LEAD_TIME_HOURS',y='Delta',
   alpha=.01)
plt.show()

# COMMAND ----------

x2.loc[((x2['Delta']>1.5) | (x2['Delta']<-1.5))]

# &(x2['LEAD_TIME_HOURS']==0)

# COMMAND ----------

x2['Delta'] = x2['FILL_RATE']-x2['prediction']
sns.scatterplot(data=x2, x='LEAD_TIME_HOURS',y='Delta',
   alpha=.01)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Importance Plot

# COMMAND ----------

import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"

feature_importance = FeatureImportance(final_model)
feature_importance.plot(top_n_features=25)


# COMMAND ----------

# from ydata_profiling import ProfileReport
# df_profile = ProfileReport(df5, minimal=False, title="Profiling Report", progress_bar=False, infer_dtypes=False)
# profile_html = df_profile.to_html()
# displayHTML(profile_html)

# COMMAND ----------


# df_profile = ProfileReport(df5, minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
# profile_html = df_profile.to_html()
# displayHTML(profile_html)

# COMMAND ----------

# from explainerdashboard import RegressionExplainer, ExplainerDashboard
# db = ExplainerDashboard(RegressionExplainer(final_model, X_test, y_test))

# db.save_html('dashboard.html')

# COMMAND ----------

# def cluster_viz(clustering_array, X_train, y_pred, optimal_k_cluster, km, active_timeframe_ids, start_date, end_date, kpi_norm_max, kpi_norm_min):
#   fig = make_subplots(rows=optimal_k_cluster, cols=1, shared_xaxes=True,
#                     column_titles=[f'{KPI_COL.replace("_", " ").title()} Trends'],
#                     row_titles=[f"Cluster {cluster+1}" for cluster in range(optimal_k_cluster)],
#                     vertical_spacing=0.05)
  
#   for cluster in range(optimal_k_cluster):

#     if ID_COL == 'account_unit_code':
#       id_idxs = np.where(y_pred==cluster)[0] 
#     elif ID_COL == 'customer_id':
#       #random sample ~5% of customers since plotting all (~6k) will just crash the viz
#       id_idxs = np.random.choice(np.where(y_pred==cluster)[0], int(len(np.where(y_pred==cluster)[0]) * 0.05)).tolist()
#       id_idxs.sort()
    
#     cluster_ids = [active_timeframe_ids[idx] for idx in id_idxs]
#     cluster_kpis = clustering_array[id_idxs]
#     cluster_xtrain = X_train[id_idxs]
#     id_array = [[x]*len(dates) for x in cluster_ids]

#     if cluster == 0:
#       showlegend = True
#     else:
#       showlegend = False

#     for idx, (norm_kpi, unnorm_kpi, id_type) in enumerate(zip(cluster_xtrain, cluster_kpis, id_array)):
#       if (cluster == 0) & (id_type[0] == id_array[0][0]):
#         breaknow = True
#       else:
#         breaknow = False
      
#       fig.add_trace(go.Scatter(name=f'{ID_COL.replace("_", " ").title()}', 
#                               x=dates, 
#                               y=norm_kpi.ravel(), 
#                               mode='lines', marker_color='mediumpurple', opacity=0.08, 
#                               showlegend=breaknow, legendgroup=f"{ID_COL.replace('_', ' ').title()}",
#                               text = [f"{ID_COL.replace('_', ' ').title()}: {i}<br>Date: {j}<br>{KPI_COL.replace('_', ' ').title()}: {k}<br>{KPI_COL.replace('_', ' ').title()} normalized: {round(l[0], 2)}" for i,j,k,l in zip(id_type, dates, unnorm_kpi, norm_kpi)]
#                               ), row=cluster+1, col=1)
    
#     fig.add_trace(go.Scatter(name=f'{ID_COL.replace("_", " ").title()} Centroid', 
#                             x=dates,
#                             y=km.cluster_centers_[cluster].ravel(), legendgroup='Centroid',
#                             mode='lines', marker_color='aquamarine', line=dict(width=3.5),
#                             showlegend=showlegend), row=cluster+1, col=1)

#   for cluster in range(optimal_k_cluster):
#     id_idxs = np.where(y_pred==cluster)[0] #indexes where cluster=1, these would be columns on aoa_array, or indexes of the id_type list aoa_id_typees
#     cluster_ids = [active_timeframe_ids[idx] for idx in id_idxs]
#     fig.update_yaxes(range=[kpi_norm_min, kpi_norm_max], row=cluster+1, col=1)
#     fig.add_annotation(x=dates[-4], y=kpi_norm_min * 0.93, showarrow=False, 
#                       text=f"n={len(cluster_ids)}", row=cluster+1, col=1)

#   fig.update_layout(title=f"EmployBridge {ID_COL.replace('_', ' ').title()} Clustered by {KPI_COL.replace('_', ' ').title()} Behavior<br><sup>Commercial Only, Timeframe: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}</sup>", autosize=False,
#                     width=1500, height=optimal_k_cluster*300, legend_tracegroupgap=25)
#   fig.update_yaxes(title=f"{KPI_COL.replace('_', ' ').title()} Normalized Average", title_standoff=0.5, row=round(optimal_k_cluster / 2), col=1)
#   # fig.write_html(f"/dbfs/FileStore/shared_uploads/{username}@employbridge.com/clustering_{KPI_COL}_weekly_euc_{START_DATE}_{END_DATE}_{seed}_{timestamp}.html")
#   # fig.show()
#   return fig

# COMMAND ----------

import plotly.graph_objects as go
from random import sample
from pyspark.sql import functions as F
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta


# COMMAND ----------

#   fig = make_subplots(rows=optimal_k_cluster, cols=1, shared_xaxes=True,
#                     column_titles=[f'{KPI_COL.replace("_", " ").title()} Trends'],
#                     row_titles=[f"Cluster {cluster+1}" for cluster in range(optimal_k_cluster)],
#                     vertical_spacing=0.05)
  
#   for cluster in range(optimal_k_cluster):

#     if ID_COL == 'account_unit_code':
#       id_idxs = np.where(y_pred==cluster)[0] 
#     elif ID_COL == 'customer_id':
#       #random sample ~5% of customers since plotting all (~6k) will just crash the viz
#       id_idxs = np.random.choice(np.where(y_pred==cluster)[0], int(len(np.where(y_pred==cluster)[0]) * 0.05)).tolist()
#       id_idxs.sort()
    
#     cluster_ids = [active_timeframe_ids[idx] for idx in id_idxs]
#     cluster_kpis = clustering_array[id_idxs]
#     cluster_xtrain = X_train[id_idxs]
#     id_array = [[x]*len(dates) for x in cluster_ids]

#     if cluster == 0:
#       showlegend = True
#     else:
#       showlegend = False

#     for idx, (norm_kpi, unnorm_kpi, id_type) in enumerate(zip(cluster_xtrain, cluster_kpis, id_array)):
#       if (cluster == 0) & (id_type[0] == id_array[0][0]):
#         breaknow = True
#       else:
#         breaknow = False
      
#       fig.add_trace(go.Scatter(name=f'{ID_COL.replace("_", " ").title()}', 
#                               x=dates, 
#                               y=norm_kpi.ravel(), 
#                               mode='lines', marker_color='mediumpurple', opacity=0.08, 
#                               showlegend=breaknow, legendgroup=f"{ID_COL.replace('_', ' ').title()}

# COMMAND ----------

import plotly.graph_objects as go
from random import sample
from pyspark.sql import functions as F
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta

# COMMAND ----------

optimal_k_cluster = 1
fig = make_subplots(rows=optimal_k_cluster, cols=1, shared_xaxes=True,
                  vertical_spacing=0.05)

# for cluster in range(optimal_k_cluster):

#   if ID_COL == 'account_unit_code':
#     id_idxs = np.where(y_pred==cluster)[0] 
#   elif ID_COL == 'customer_id':
#     #random sample ~5% of customers since plotting all (~6k) will just crash the viz
#     id_idxs = np.random.choice(np.where(y_pred==cluster)[0], int(len(np.where(y_pred==cluster)[0]) * 0.05)).tolist()
#     id_idxs.sort()
  
#   cluster_ids = [active_timeframe_ids[idx] for idx in id_idxs]
#   cluster_kpis = clustering_array[id_idxs]
#   cluster_xtrain = X_train[id_idxs]
#   id_array = [[x]*len(dates) for x in cluster_ids]

#   if cluster == 0:
#     showlegend = True
#   else:
#     showlegend = False

#   for idx, (norm_kpi, unnorm_kpi, id_type) in enumerate(zip(cluster_xtrain, cluster_kpis, id_array)):
#     if (cluster == 0) & (id_type[0] == id_array[0][0]):
#       breaknow = True
#     else:
#       breaknow = False
    
#     fig.add_trace(go.Scatter(name=f'{ID_COL.replace("_", " ").title()}', 
#                             x=dates, 
#                             y=norm_kpi.ravel(), 
#                             mode='lines', marker_color='mediumpurple', opacity=0.08, 
#                             showlegend=breaknow, legendgroup=f"{ID_COL.replace('_', ' ').title()}
for job_id in set(x2['JOB_ID']):
  if np.random.rand() < .5:
    fig.add_trace(
        go.Scatter(
            x=np.array(x2.loc[x2['JOB_ID']==job_id,'normalized_lead_time']),
            y=np.array(x2.loc[x2['JOB_ID']==job_id,'current_fill_rate']),
            mode="markers+lines",
            marker_color='mediumpurple', opacity=0.02, 
            showlegend=False)
    )
fig.update_layout(
  height = 1000,
  width = 1000
)
fig.show()

# COMMAND ----------


