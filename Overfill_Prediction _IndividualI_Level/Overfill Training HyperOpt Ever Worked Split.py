# Databricks notebook source
# MAGIC %md
# MAGIC # Individual Level Model Description
# MAGIC This notebook runs util notebooks, queries the foreign catalog to obtain individual level application event data, cleans it, and develops a model to predict individual-level show up rates.  It answers the question, "if someone applied, what was the outcome of that application?" by first checking to see if they worked the first shift of the applied to job.  If not, it checks when their latest cancellation occured and categorizes them into early cancellation, SNC, and NCNS.  This model will supplement the job-level overfill prediction model and apply its predictions to the applicant population.

# COMMAND ----------

# MAGIC %md
# MAGIC # Library installs and util runs

# COMMAND ----------

# %pip install databricks-feature-engineering
%pip install databricks-feature-engineering==0.2.1a1
%pip install explainerdashboard
%pip install ydata_profiling
%pip install kds
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/ml-modeling-utils
# MAGIC

# COMMAND ----------

# MAGIC %run ./_resources/mlflow-utils
# MAGIC

# COMMAND ----------

# MAGIC %run ./_resources/overfill-utils

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
from xgboost import XGBClassifier

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking import MlflowClient


# COMMAND ----------

# MAGIC %md
# MAGIC # Training Label Query

# COMMAND ----------

start_date = '2023-01-01'
now = datetime.now()
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
# end_date = '2024-01-01'

sdf = jobs_query(start_date,end_date)
# display(sdf)


# COMMAND ----------

# MAGIC %md
# MAGIC # Filtering and cleaning

# COMMAND ----------

#currently only trust training data for BC accounts where the account was posted in advance of starting and requiring more than 0 people.
sdf = sdf.filter((sdf.NEEDED >0)&(sdf.COMPANY_ORIGIN == 'BC'))

#changes the categorical target variable into a binary one-versus-rest status for worked vs. all other cancellation taypes and NCNS
sdf = sdf.withColumn('Work', F.when(sdf.target_var == 'Worked', 1).otherwise(0))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Foreign Catalog to Delta Temp Table
# MAGIC ##### We have found problems with querying the foreign catalog and then looking up features.  It seems to take 10-20x as long compared to writing a temp Delta table, rereading it, and then performing the feature lookup.

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bluecrew.ml.overfill_training_labels

# COMMAND ----------

from pyspark.sql import DataFrame

def write_spark_table_to_databricks_schema(df: DataFrame, table_name: str, schema_name: str = 'bluecrew.ml', mode: str = 'overwrite'):
    """
    Write a Spark DataFrame to a table within a specific schema in Databricks.

    Parameters:
    - df: The Spark DataFrame to write.
    - table_name: The name of the target table.
    - schema_name: The name of the schema (database) in Databricks. Default is 'bluecrew.ml'.
    - mode: Specifies the behavior when the table already exists. Options include:
      - 'append': Add the data to the existing table.
      - 'overwrite': Overwrite the existing table.
      - 'ignore': Silently ignore this operation if the table already exists.
      - 'error' or 'errorifexists': Throw an exception if the table already exists.
    """
    # Define the full table path including the schema name
    full_table_name = f"{schema_name}.{table_name}"
    
    # Write the DataFrame to the table in the specified schema
    df.write.mode(mode).saveAsTable(full_table_name)

    print(f"DataFrame written to table {full_table_name} in mode '{mode}'.")



write_spark_table_to_databricks_schema(optimize_spark(sdf), 'overfill_training_labels', 'bluecrew.ml')


# COMMAND ----------

sdf = spark.read.format("delta").table('bluecrew.ml.overfill_training_labels')

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Store Lookup

# COMMAND ----------

# This looks up job-level features in feature store
# # For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html
fe = FeatureEngineeringClient()
model_feature_lookups = [
      #This is a feature lookup that demonstrates how to use point in time based lookups for training sets
      FeatureLookup(
        table_name='feature_store.dev.jobs_data',
        feature_names=['COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'WAGE_DELTA', 'SCHEDULE_NAME_UPDATED', "COUNTY", "JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE", 'JOB_TYPE', 'JOB_WAGE', 'JOB_TITLE', 'JOB_OVERFILL', 'JOB_SHIFTS', 'INVITED_WORKER_COUNT', 'POSTING_LEAD_TIME_DAYS'],
        lookup_key="JOB_ID",
        timestamp_lookup_key="min_successful_app_start"),
      FeatureLookup(
        table_name='feature_store.dev.user_hours_worked_calendar2',
        lookup_key="USER_ID",
        timestamp_lookup_key="min_successful_app_start"),
      FeatureLookup(
        table_name='feature_store.dev.user_snc_ncns_calendar',
        lookup_key="USER_ID",
        feature_names=['NCNS_SHIFTS_TOTAL', 'NCNS_SHIFTS_LAST_30_DAYS', 'NCNS_SHIFTS_LAST_90_DAYS', 'NCNS_SHIFTS_LAST_365_DAYS', 'SNC_SHIFTS_TOTAL', 'SNC_SHIFTS_LAST_30_DAYS', 'SNC_SHIFTS_LAST_90_DAYS', 'SNC_SHIFTS_LAST_365_DAYS'],
        timestamp_lookup_key="min_successful_app_start"),
      # FeatureLookup(
      #   table_name='feature_store.dev.job_schedule_array',
      #   lookup_key='JOB_ID',
      #   timestamp_lookup_key="min_successful_app_start"
      # ),
      # FeatureLookup(
      #   table_name='feature_store.dev.user_schedule_array2',
      #   lookup_key="USER_ID",
      #   timestamp_lookup_key="min_successful_app_start"
      # ),
      FeatureLookup(
        table_name='feature_store.dev.user_work_history2',
        lookup_key="USER_ID",
        timestamp_lookup_key="min_successful_app_start"
      ), 
      # FeatureLookup(
      #   table_name='feature_store.dev.user_funnel_timeline',
      #   lookup_key="USER_ID",
      #   feature_names=['ever_worked'],
      #   timestamp_lookup_key="min_successful_app_start"
      # ),
      # FeatureLookup(
      #   table_name='feature_store.dev.direct_invites',
      #   lookup_key=["USER_ID","JOB_ID"],
      #   # feature_names=['ever_worked'],
      #   # timestamp_lookup_key="min_successful_app_start"
      # ), 
      FeatureLookup(
        table_name='feature_store.dev.cm_quality_attendancev0',
        lookup_key="USER_ID",
        feature_names=['POTENTIAL_SHIFTS_LOST', 'SHIFT_DOLLARS_GAINED', 'SHIFT_DOLLARS_LOST', 'SHIFT_DOLLARS_NET', 'CM_ECONOMIC_PROFIT', 'CM_NET_SCORE', 'SINGLE_SHIFT_CANCELS_LAST_13WEEKS', 'SINGLE_SHIFT_SNC_LAST_13WEEKS','SINGLE_SHIFT_NCNS_LAST_13WEEKS', 'POSTJOBSTART_QUITS_LAST_13WEEKS', 'PREJOB_CANCELS_LAST_13WEEKS', 'PREJOB_SNC_LAST_13WEEKS', 'PREJOB_NCNS_LAST_13WEEKS', 'SHIFTS_LAST_13WEEKS', 'CLUSTER_FINAL'],
        timestamp_lookup_key="min_successful_app_start"
      ), 
      # FeatureLookup(
      #   table_name='feature_store.dev.user_certs',
      #   lookup_key="USER_ID",
      #   timestamp_lookup_key="min_successful_app_start"
      # ), 
      FeatureLookup(
        table_name='feature_store.dev.user_ratings',
        lookup_key="USER_ID",
        timestamp_lookup_key="min_successful_app_start"
      ), 
      # # Calculate a new feature called `cosine_sim` - the cosine similarity between the user's work history and the current job.
      # FeatureFunction(
      #   udf_name='feature_store.dev.cosine_similarity',
      #   output_name="cosine_sim",
      #   input_bindings={"arr1":"job_schedule", "arr2": "running_schedule_array"},
      # ), 
      # Calculate a new feature called `commute_distance` - the distance between a user's address and the current job address.
      FeatureFunction(
        udf_name='feature_store.dev.distance',
        output_name="commute_distance",
        input_bindings={"lat1":"user_address_latitude", "lon1": "user_address_longitude", "lat2":"JOB_ADDRESS_LATITUDE", "lon2": "JOB_ADDRESS_LONGITUDE"},
      )
]
training_set = fe.create_training_set(
    df = sdf, # joining the original Dataset, with our FeatureLookupTable
    feature_lookups=model_feature_lookups,
    exclude_columns=[], # exclude columns as we don't want them as feature
    label='Work'
)

training_pd = training_set.load_df()
# display(training_pd)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bluecrew.ml.overfill_feature_lookup

# COMMAND ----------

write_spark_table_to_databricks_schema(training_pd, 'overfill_feature_lookup', 'bluecrew.ml')


# COMMAND ----------

training_pd = spark.read.format("delta").table('bluecrew.ml.overfill_feature_lookup')

# COMMAND ----------

# MAGIC %md
# MAGIC # Pandas DataFrame Creation 

# COMMAND ----------

df = optimize_spark(training_pd).toPandas()

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  
  df[col] = df[col].astype(int)
print(list(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC # Testing and training preparation
# MAGIC ##### This has to happen in the feature lookup to be part of the fe.log_model pipeline and use the predict_batch function.  Currently, these are things I am testing in model and will clean up once I find which sets of features to use and which calculated fields are relevant.

# COMMAND ----------

# Defines the columns that correspond to a job and creates final dataset to split for training and testing.
cols_to_drop = ['JOB_STATUS_ENUM', 'JOB_STATUS', 'JOB_OVERFILL', 'INVITED_WORKER_COUNT', 'SEGMENT_INDEX', 'NEEDED', 'POSITION_ID', 'COMPANY_ID', 'SCHEDULE_ID', 'JOB_ID','USER_ID', 'JOB_CITY','COUNTY','target_var', 'application_status', 'successful_application_count', 'max_successful_app_start', 'min_successful_app_end', 'max_successful_app_end', 'JOB_TITLE', 'JOB_STATE', 'COMPANY_ORIGIN']
# df2 = df[(df['target_var']!='Early Cancel')&(df['JOB_STATUS']!='Cancelled')]
# df2 = df[(df['JOB_STATUS']!='Cancelled')]
# df2 = df[(df['target_var'] !='Early Cancel')&(df['target_var'] != 'SNC')&(df['JOB_STATUS']!='Cancelled')&(df['JOB_CREATED_AT']>='2023-01-01')& (df['POSTING_LEAD_TIME_DAYS']>0)&(df['commute_distance']<=70)&(df['ever_worked']==1)]
df2 = df[(df['target_var'] !='Early Cancel')&(df['target_var'] != 'SNC')&(df['JOB_STATUS']!='Cancelled')&(df['JOB_CREATED_AT']>='2023-01-01')& (df['POSTING_LEAD_TIME_DAYS']>0)&(df['commute_distance']<=70)]
df2['CLUSTER_FINAL']= df2['CLUSTER_FINAL'].fillna(-1).astype(int).astype(str)
# df2 = df.copy()
df2['Wage_diff']=df2['JOB_WAGE']/df2['avg_wage_total']
df2['ever_worked']=df2['jobs_worked_total'].apply(lambda x: 1 if x>0 else 0)
df2['Unique_ID']= np.arange(df2.shape[0])
df3 = df2[['Unique_ID', 'USER_ID', 'JOB_ID']]
df4 = df2.drop(columns=cols_to_drop)
df5 = df4.set_index('Unique_ID')
# df5['Past_Work']= df5['cosine_sim'].apply(lambda x: 1 if x==x else 0)
df_prior_worked = df5[df5['ever_worked']==1]
df_never_worked = df5[df5['ever_worked']==0]
# df5 = df5[df5['apply_lead_time_hours']!=0]
df5.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Prior Work Model Training

# COMMAND ----------

mlflow.autolog()
mlflow.set_registry_uri('databricks-uc')
mlflow.set_tracking_uri("databricks")
# mlflow.set_experiment("antwoineflowers/databricks-uc/jobmatchscoring")
model_name = 'overfill_test'

 # Separate features and target variable
X = df_prior_worked.drop('Work', axis=1)
y = df_prior_worked['Work']

# Create Preprocessing Pipeline

# Identify numerical and categorical columns (excluding the label column)
numerical_cols = df_prior_worked.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.drop('Work')

categorical_cols = df_prior_worked.select_dtypes(include=['object', 'category']).columns

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
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=9986, stratify=y )
X_val, X_test_prior, y_val, y_test_prior = train_test_split(X_rem, y_rem, test_size=0.5, random_state=72688, stratify=y_rem)

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
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
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

        # Initialize XGBClassifier
        model = XGBClassifier(**params 
        )

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Predict and evaluate using the probabilities of the positive class
        predictions_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate metrics using probabilities
        auc_score = roc_auc_score(y_val, predictions_proba)
        precision, recall, _ = precision_recall_curve(y_val, predictions_proba)
        f1 = f1_score(y_val, (predictions_proba > 0.5).astype(int))
        
        # Convert probabilities to binary outcomes based on a 0.5 cutoff
        predictions_binary = (predictions_proba > 0.5).astype(int)

        return {'loss': -auc_score, 'status': STATUS_OK, 'model': pipeline, 'precision': precision, 'f1_score': f1, 'recall':recall}

# Hyperparameter optimization with Hyperopt
with mlflow.start_run(run_name='Hyperparameter Optimization') as parent_run:
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,  
        algo=tpe.suggest,
        max_evals=10,  
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
auc_roc = -1*model['loss']



# COMMAND ----------

# # Define the model name for the registry
registry_model_name = "bluecrew.ml.overfill_prior_worked"

client = MlflowClient()
# client.delete_registered_model(name=registry_model_name)

latest_experiment = find_latest_experiment()
best_run_id = find_best_run_id(latest_experiment, "metrics.training_roc_auc")
# Uncomment this to register the model
model_details = register_run_as_model(registry_model_name, best_run_id)
# update_model_stage(model_details, 'Staging')

# COMMAND ----------

# MAGIC %md
# MAGIC # Never Worked Model Training

# COMMAND ----------

mlflow.autolog()
mlflow.set_registry_uri('databricks-uc')
mlflow.set_tracking_uri("databricks")
# mlflow.set_experiment("antwoineflowers/databricks-uc/jobmatchscoring")
model_name = 'overfill_test'

 # Separate features and target variable
X = df_never_worked.drop('Work', axis=1)
y = df_never_worked['Work']

# Create Preprocessing Pipeline

# Identify numerical and categorical columns (excluding the label column)
numerical_cols = df_never_worked.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.drop('Work')

categorical_cols = df_never_worked.select_dtypes(include=['object', 'category']).columns

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
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=9986, stratify=y )
X_val, X_test_never, y_val, y_test_never = train_test_split(X_rem, y_rem, test_size=0.5, random_state=72688, stratify=y_rem)

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
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
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

        # Initialize XGBClassifier
        model = XGBClassifier(**params 
        )

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Predict and evaluate using the probabilities of the positive class
        predictions_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate metrics using probabilities
        auc_score = roc_auc_score(y_val, predictions_proba)
        precision, recall, _ = precision_recall_curve(y_val, predictions_proba)
        f1 = f1_score(y_val, (predictions_proba > 0.5).astype(int))
        
        # Convert probabilities to binary outcomes based on a 0.5 cutoff
        predictions_binary = (predictions_proba > 0.5).astype(int)

        return {'loss': -auc_score, 'status': STATUS_OK, 'model': pipeline, 'precision': precision, 'f1_score': f1, 'recall':recall}

# Hyperparameter optimization with Hyperopt
with mlflow.start_run(run_name='Hyperparameter Optimization') as parent_run:
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,  
        algo=tpe.suggest,
        max_evals=10,  
        trials=trials
    )

# Fetch the details of the best run
best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
best_params = space_eval(space, best)

# Print out the best parameters and corresponding loss
print(f"Best parameters: {best_params}")
print(f"Best eval auc: {best_run['loss']}")


# COMMAND ----------

mlflow.sklearn.autolog()
# Log the best hyperparameters
with mlflow.start_run(run_name='best_model_run'):
  model = objective(best_params)
  mlflow.set_tag("mlflow.best_model_run", "best_run")
final_model = model['model']  
auc_roc = -1*model['loss']



# COMMAND ----------

# # Define the model name for the registry
registry_model_name2 = "bluecrew.ml.overfill_never_worked"

client = MlflowClient()
# client.delete_registered_model(name=registry_model_name)

latest_experiment = find_latest_experiment()
best_run_id = find_best_run_id(latest_experiment, "metrics.training_roc_auc")
# Uncomment this to register the model
model_details = register_run_as_model(registry_model_name2, best_run_id)
# update_model_stage(model_details, 'Staging')

# COMMAND ----------

# MAGIC %md
# MAGIC # Sample code to show model loading

# COMMAND ----------

#this doesn't come with predict_proba.  I think you could build a custom model wrapper to fix this, but I didn't.
mlflow.set_registry_uri('databricks-uc')
registry_model_name = "bluecrew.ml.overfill_prior_worked"
registry_model_name2 = "bluecrew.ml.overfill_never_worked"
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{registry_model_name}/1")
#this allows you to get the predict_proba method
prior_work_model = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name}/1")
never_work_model = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name2}/1")


# pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{registry_model_name}/1")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Check model Signature
# import mlflow
# model_uri = client.get_model_version_download_uri('toy-model','10')
prior_work_model_info = mlflow.models.get_model_info(f"models:/{registry_model_name}/1")
prior_work_sig_dict = prior_work_model_info._signature_dict
prior_work_sig_dict['inputs']

never_work_model_info = mlflow.models.get_model_info(f"models:/{registry_model_name2}/1")
never_work_sig_dict = prior_work_model_info._signature_dict
never_work_sig_dict['inputs']


# COMMAND ----------

import ast
true = True
false = False
prior_worked_schema_dict = {}
prior_work_input_dict_list = eval(prior_work_sig_dict['inputs'])
# print(input_dict)
for value in prior_work_input_dict_list:
  # print(value)
  prior_worked_schema_dict[value['name']]=value['type']
prior_worked_schema_dict

never_worked_schema_dict = {}
never_work_input_dict_list = eval(never_work_sig_dict['inputs'])
# print(input_dict)
for value in never_work_input_dict_list:
  # print(value)
  never_worked_schema_dict[value['name']]=value['type']
never_worked_schema_dict

# COMMAND ----------

# col_list = [key for key in schema_dict]
# print(col_list)
df_prior_work_test = pd.concat([X_test_prior.reset_index(),y_test_prior.reset_index().drop(columns = ['Unique_ID'])], axis = 1)
# df5[df5['ever_worked']==1]
df_never_work_test = pd.concat([X_test_never.reset_index(),y_test_never.reset_index().drop(columns = ['Unique_ID'])], axis = 1)

for key in prior_worked_schema_dict:
  if df_prior_work_test.dtypes[key]==prior_worked_schema_dict[key]:
    # print('ok')
    pass
  else:
    print(f"df_type: {df_prior_work_test.dtypes[key]}, schema_req: {prior_worked_schema_dict[key]}")
    # df_test[key].astype(schema_dict[key])

df_prior_work_test['predicted_prob'] = prior_work_model.predict_proba(df_prior_work_test[[key for key in prior_worked_schema_dict]])[:,1]
df_prior_work_test['prediction'] = prior_work_model.predict(df_prior_work_test[[key for key in prior_worked_schema_dict]])

for key in never_worked_schema_dict:
  if df_never_work_test.dtypes[key]==never_worked_schema_dict[key]:
    # print('ok')
    pass
  else:
    print(f"df_type: {df_never_work_test.dtypes[key]}, schema_req: {never_worked_schema_dict[key]}")
    # df_test[key].astype(schema_dict[key])

df_never_work_test['predicted_prob'] = never_work_model.predict_proba(df_never_work_test[[key for key in never_worked_schema_dict]])[:,1]
df_never_work_test['prediction'] = never_work_model.predict(df_never_work_test[[key for key in never_worked_schema_dict]])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction analysis

# COMMAND ----------

prediction_analysis_df = pd.concat([df_prior_work_test, df_never_work_test])
prediction_analysis_df

# COMMAND ----------

# # y2 = df5['Work'].reset_index()
# # Creating a dictionary by passing Series objects as values
# # y2['Predictions']=predictions
# x2 = pd.concat([X_test.reset_index(),predictions,y_test.reset_index()['Work']],axis = 1)
# x2.rename(columns={0:'predicted_probability'}, inplace = True)
# # y3 = y2.merge(X.reset_index(), on = 'Unique_ID',how = 'inner')
# display(x2)

# x2.groupby(['Work']).count()
 

# COMMAND ----------

sns.kdeplot(
   data=prediction_analysis_df, x='predicted_prob', hue="Work",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
plt.show()

# COMMAND ----------

# x2['ever_worked'].fillna(0,inplace = True)
sns.kdeplot(
   data=prediction_analysis_df, x='predicted_prob', hue="ever_worked",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
plt.show()

# COMMAND ----------

# sns.histplot(data=x2[x2['CLUSTER_FINAL']!= '-1'], x='predicted_probability', hue="CLUSTER_FINAL",palette="Spectral",
#    alpha=.5, linewidth=0,bins=10, multiple='stack')
# plt.show()

# COMMAND ----------

sns.histplot(data=prediction_analysis_df, x='predicted_prob', hue="Work",palette="Spectral",
   alpha=.5, linewidth=0,bins=10, multiple='stack')
plt.show()

# COMMAND ----------

jobs_and_users = prediction_analysis_df.merge(df3, on = 'Unique_ID', how = 'left')
job_modeling = jobs_and_users[['JOB_ID','predicted_prob','Work']].groupby('JOB_ID').agg({'Work': 'sum', 'JOB_ID':'count', 'predicted_prob': 'sum'})
job_modeling['work_rate']= job_modeling['Work']/job_modeling['JOB_ID']
job_modeling['predicted_work_rate']= job_modeling['predicted_prob']/job_modeling['JOB_ID']
job_modeling2 = job_modeling[job_modeling['JOB_ID']>3]
# display(job_modeling)

# COMMAND ----------

(rmse, mae, r2, mape) = eval_metrics(job_modeling['Work'], job_modeling['predicted_prob'])

# Print out model metrics
print("Comparing expected show ups:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

(rmse, mae, r2, mape) = eval_metrics(job_modeling['work_rate'], job_modeling['predicted_work_rate'])

# Print out model metrics
print("Comparing Work Rate:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

(rmse, mae, r2, mape) = eval_metrics(job_modeling2['work_rate'], job_modeling2['predicted_work_rate'])

# Print out model metrics
print("Comparing Work Rate for jobs with 5+ signups:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

# COMMAND ----------

import kds

kds.metrics.report(job_modeling['Work'], job_modeling['predicted_prob'])
plt.show()


# COMMAND ----------

sns.scatterplot(data=job_modeling, x='Work',y='predicted_prob',
   alpha=.1)
plt.show()

# COMMAND ----------

sns.scatterplot(data=prediction_analysis_df, x='jobs_worked_total',y='predicted_prob',
   alpha=.05, hue='Work')
plt.show()

# COMMAND ----------

sns.scatterplot(data=job_modeling2, x='work_rate',y='predicted_work_rate',
   alpha=.5)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Importance Plot

# COMMAND ----------

import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"

feature_importance = FeatureImportance(prior_work_model)
feature_importance.plot(top_n_features=127)


# COMMAND ----------

feature_importance = FeatureImportance(never_work_model)
feature_importance.plot(top_n_features=127)

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

# from explainerdashboard import ClassifierExplainer, ExplainerDashboard
# db = ExplainerDashboard(ClassifierExplainer(final_model, X_test, y_test))

# db.save_html('dashboard.html')

# COMMAND ----------


