# Databricks notebook source
# MAGIC %md
# MAGIC # Job Level Model
# MAGIC This notebook loads snowflake data, cleans it, and develops a model to predict job-level show up rates (worked first shift/successful applications at job start) that will help with setting an overfill rate.  This model is at the job level and doesn't consider the attricutes of CMs that have applied.  Since it only considers successful applications at the time of job start, all cancellations prior to job start are excluded.

# COMMAND ----------

# MAGIC %md
# MAGIC #Preprocessing

# COMMAND ----------

# %pip install databricks-feature-engineering
%pip install databricks-feature-engineering==0.2.1a1
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
from pyspark.sql.functions import to_date, current_timestamp
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import mlflow
from mlflow.models.signature import infer_signature


# COMMAND ----------

now = datetime.now()
start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
end_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")

sdf = jobs_query(start_date,end_date)
display(sdf)



# COMMAND ----------

#currently only trust training data for BC accounts where the account was posted in advance of starting and requiring more than 0 people.
# sdf = sdf.filter((sdf.NEEDED >0))



# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bluecrew.ml.overfill_prediction_labels

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



write_spark_table_to_databricks_schema(optimize_spark(sdf), 'overfill_prediction_labels', 'bluecrew.ml')


# COMMAND ----------

sdf = spark.read.format("delta").table('bluecrew.ml.overfill_prediction_labels')

# COMMAND ----------

# This looks up job-level features in feature store
# # For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html
fe = FeatureEngineeringClient()
model_feature_lookups = [
      #This is a feature lookup that demonstrates how to use point in time based lookups for training sets
      FeatureLookup(
        table_name='feature_store.dev.jobs_data',
        # feature_names=['COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'WAGE_DELTA', 'SCHEDULE_NAME_UPDATED', "COUNTY", "JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE", 'JOB_TYPE', 'JOB_WAGE', 'JOB_TITLE', 'JOB_OVERFILL', 'JOB_SHIFTS', 'INVITED_WORKER_COUNT', 'POSTING_LEAD_TIME_DAYS'],
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
    label=None
)

training_pd = training_set.load_df()
# display(training_pd)

# COMMAND ----------

# Converts the Spark df + features to a pandas DataFrame and turns boolean columns into integers
# convert this to do the type casting first and then use toPandas()?
df = optimize_spark(training_pd).toPandas()

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))
df.set_index(['JOB_ID','USER_ID'], inplace = True)

# COMMAND ----------

# df['Wage_diff_7']=df['JOB_WAGE']/df['avg_wage_last_7_days']
df['Wage_diff']=df['JOB_WAGE']/df['avg_wage_total']
df['ever_worked']=df['jobs_worked_total'].apply(lambda x: 1 if x>0 else 0)
df['CLUSTER_FINAL']= df['CLUSTER_FINAL'].fillna(-1).astype(int).astype(str)

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
# Define the model name for the registry
registry_model_name = "bluecrew.ml.overfill_test"

model2 = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name}/1")
model_info = mlflow.models.get_model_info(f"models:/{registry_model_name}/1")
sig_dict = model_info._signature_dict
# sig_dict['inputs']




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

df2 = df.copy()
df2['predicted_prob'] = model2.predict_proba(df2[[key for key in schema_dict]])[:,1]
df2['prediction'] = model2.predict(df2[[key for key in schema_dict]])

# COMMAND ----------

now = datetime.now()
df2['as_of_date'] = now
pred_df = spark.createDataFrame(df2.reset_index())

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bluecrew.ml.individual_overfill_prediction_output

# COMMAND ----------

write_spark_table_to_databricks_schema(optimize_spark(pred_df), 'individual_overfill_prediction_output', 'bluecrew.ml', mode = 'append')

# COMMAND ----------

# MAGIC %md
# MAGIC # Need to add the post processing outputs to get to new predictions for Streamlit

# COMMAND ----------

sdf2 = spark.createDataFrame(df2.reset_index())
sdf2.createOrReplaceTempView('data')

# COMMAND ----------

sdf2.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL_INDIVIDUAL_PREDICTIONS').save()


# COMMAND ----------

plt.figure(figsize=(15, 10))

sns.histplot(data=df2, x='predicted_prob', hue="JOB_TYPE",palette="tab20",
   alpha=.5, linewidth=0,bins=10, multiple='stack')


# sns.histplot(data=df2, x='predicted_prob', hue="JOB_TYPE",palette="Spectral",
#    alpha=.5, linewidth=0,bins=10, multiple='stack')
plt.show()

# COMMAND ----------

