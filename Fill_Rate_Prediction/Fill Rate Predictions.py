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
end_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
# end_date = '2024-01-01'


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
    -- AND jsw.job_start_date_time > DATEADD(HOUR, -1, SYSDATE())
    AND shift_sequence = 1
    AND jsw.job_start_date_time <= shift_start_time
    AND jsw.job_created_at <= shift_start_time
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
    datediff(HOUR, calendar_date, max_date) as lead_time_hours
FROM calendar_hours
INNER JOIN job_dates j
ON calendar_date >=date_trunc(MINUTE,min_date)
AND calendar_date <= max_date
and (calendar_date = dateadd(MINUTE,1, date_trunc(MINUTE,min_date)) or MINUTE(calendar_date) = 0)
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
      #Lookup overfill
      FeatureLookup(
        table_name='feature_store.dev.job_overfill_change',
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
    label=None
)

training_pd = training_set.load_df()
# display(training_pd2)
# display(training_pd)

# COMMAND ----------

# Converts the Spark df + features to a pandas DataFrame and turns boolean columns into integers
# convert this to do the type casting first and then use toPandas()?
df = optimize_spark(training_pd).toPandas()

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))
# df.set_index(['JOB_ID'], inplace = True)

# COMMAND ----------

# Defines the columns that correspond to a job and creates final dataset to split for training and testing.
cols_to_drop = ['MIN_DATE', 'MAX_DATE', 'FINAL_NEEDED', 'JOB_OVERFILL', 'JOB_TEMPLATE_ID', 'EXTERNAL_JOB_ID', 'JOB_TEMPLATE_ID', 'JOB_TEMPLATES_EXTERNAL_ID', 'BLUESHIFT_REQUEST_ID', 'FORKED_FROM', 'JOB_POSTER_ID', 'JOB_POSTER_FULL_NAME', 'JOB_POSTER_EMAIL', 'JOB_TYPE_ID', 'JOB_DESCRIPTION', 'SCHEDULE_ID', 'SCHEDULE_NAME', 'JOB_STATUS_ENUM', 'NOTES_JOBS_DATA', 'JOB_BATCH_SIZE', 'JOB_REASON_CODE', 'JOB_REASON_TEXT', 'JOB_ADDRESS', 'JOB_ADDRESS_LINE_TWO', 'JOB_CITY', 'COUNTY','JOB_TITLE', 'JOB_ZIPCODE', 'JOB_ADDRESS_LATITUDE', 'JOB_ADDRESS_LONGITUDE', 'JOB_REGION_ID', 'FILL_RATE']
# df2 = df[(df['target_var']!='Early Cancel')&(df['JOB_STATUS']!='Cancelled')]
df2 = df.copy()
df2['total_lead_time'] = df2.groupby('JOB_ID')['LEAD_TIME_HOURS'].transform('max')
df2['normalized_lead_time']=df2.apply(lambda x: 1 - x['LEAD_TIME_HOURS']/(x['total_lead_time'] if x['total_lead_time']>0 else 1),axis = 1)
df2['needed_and_overfill'] = df2['NEEDED']+df2['OVERFILL']
df2['blueshift'] = df2['BLUESHIFT_REQUEST_ID'].apply(lambda x: 1 if x==x else 0)
df2['forked'] = df2['FORKED_FROM'].apply(lambda x: 1 if x==x else 0)
df2['overfill_rate']=df2.apply(lambda x: x['total_applicants']/(x['NEEDED']+x['OVERFILL']) if (x['NEEDED']+x['OVERFILL'])>0 else 0, axis = 1)
df2['POSTING_LEAD_TIME_BINS']= df2['POSTING_LEAD_TIME_DAYS'].apply(lambda x: "0-3" if x <=3 else "4-7" if x<=7 else "7-14" if x<=14 else "15-30" if x<=30 else "31+")
df4 = df2.drop(columns=cols_to_drop)
df5 = df4.set_index('JOB_ID')
df5['current_fill_rate']= df5.apply(lambda x: x['total_applicants']/x['NEEDED'] if x['NEEDED']>0 else -1, axis = 1)
df5['prior_1_fill_rate'] = df5.groupby(['JOB_ID'])['current_fill_rate'].shift(1)
df5['current_confirmed_rate']= df5.apply(lambda x: x['confirmed']/x['total_applicants'] if x['total_applicants']>0 else 0 if x['requested']>0 else -1, axis = 1)
df5['current_fill_rate'].fillna(1,inplace = True)
# df5 = df5[(df5['LEAD_TIME_HOURS']!=0)&(df5['total_applicants']>=0)&(df5['NEEDED']>1)&(df5['FILL_RATE']< 4)&(df5['POSTING_LEAD_TIME_DAYS']>0)]
# df5 = df5[(df5['total_applicants']>=0)&(df5['NEEDED']>1)&(df5['FILL_RATE']< 4)&(df5['POSTING_LEAD_TIME_DAYS']>0)&(df5['JOB_STATUS']!='Cancellled')]
df5.describe()

# COMMAND ----------

# df5[df5.index == '334419'].sort_values(by = 'CALENDAR_DATE')

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
# Define the model name for the registry
registry_model_name = "bluecrew.ml.fill_rate_test"

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

df6 = df5.copy()
# df6['predicted_prob'] = model2.predict_proba(df6[[key for key in schema_dict]])[:,1]
df6['prediction'] = model2.predict(df6[[key for key in schema_dict]])

# COMMAND ----------

now = datetime.now()
df6['as_of_date'] = now
pred_df = spark.createDataFrame(df6.reset_index())

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bluecrew.ml.fill_rate_prediction_output

# COMMAND ----------

write_spark_table_to_databricks_schema(optimize_spark(pred_df), 'fill_rate_prediction_output', 'bluecrew.ml', mode = 'append')

# COMMAND ----------

# MAGIC %md
# MAGIC # Need to add the post processing outputs to get to new predictions for Streamlit

# COMMAND ----------

# sdf2 = spark.createDataFrame(df5.reset_index())
# sdf2.createOrReplaceTempView('data')
pred_df = pred_df.drop("NOTES_JOBS_DATA", "JOB_REASON_TEXT" )

# COMMAND ----------

pred_df.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'FILL_RATE_PREDICTIONS').save()


# COMMAND ----------

# plt.figure(figsize=(15, 10))

# sns.histplot(data=df2, x='predicted_prob', hue="JOB_TYPE",palette="tab20",
#    alpha=.5, linewidth=0,bins=10, multiple='stack')


# # sns.histplot(data=df2, x='predicted_prob', hue="JOB_TYPE",palette="Spectral",
# #    alpha=.5, linewidth=0,bins=10, multiple='stack')
# plt.show()

# COMMAND ----------


