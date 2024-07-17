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
        # feature_names=['COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'JOB_TYPE_TITLE_COUNT', 'TOTAL_JOB_COUNT', 'TOTAL_CMS_REQUIRED', 'WAGE_DELTA', 'WAGE_DELTA_PERCENT', 'ELIGIBLE_USERS',  'ACTIVE_USERS_7_DAYS', 'ELIGIBLE_CMS_1_MILE', 'ELIGIBLE_CMS_5_MILE', 'ELIGIBLE_CMS_10_MILE',  'ELIGIBLE_CMS_15_MILE', 'ACTIVE_CMS_1_MILE', 'ACTIVE_CMS_5_MILE', 'ACTIVE_CMS_10_MILE', 'ACTIVE_CMS_15_MILE', 'EXTERNAL_JOB_ID', 'JOB_TEMPLATE_ID', 'JOB_TEMPLATES_EXTERNAL_ID', 'BLUESHIFT_REQUEST_ID', 'FORKED_FROM', 'JOB_POSTER_ID', 'JOB_POSTER_FULL_NAME', 'JOB_POSTER_EMAIL', 'JOB_TYPE_ID', 'JOB_START_DATE_TIME', 'JOB_END_DATE_TIME', 'JOB_TYPE', 'JOB_TITLE', 'JOB_DESCRIPTION', 'SCHEDULE_NAME', 'JOB_NEEDED_LAST_COUNT', 'JOB_WAGE', 'NOTES_JOBS_DATA', 'JOB_BATCH_SIZE', 'JOB_OVERFILL', 'JOB_DAYS', 'JOB_SHIFTS', 'INVITED_WORKER_COUNT', 'JOB_IS_APPLICATION', 'JOB_REASON_CODE', 'JOB_REASON_TEXT', 'JOB_ADDRESS', 'JOB_ADDRESS_LINE_TWO',  'JOB_ZIPCODE', 'JOB_ADDRESS_LATITUDE', 'JOB_ADDRESS_LONGITUDE', 'JOB_REGION_ID', 'JOB_REGION', 'CM_COUNT_RATIO', 'SCHEDULE_NAME_UPDATED', 'POSTING_LEAD_TIME_DAYS'],
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
        table_name='feature_store.dev.user_work_history',
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

# MAGIC %md
# MAGIC # Prior Worked Predictions

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
# Define the model name for the registry
registry_model_name = "bluecrew.ml.overfill_prior_worked"
registry_model_name2 = "bluecrew.ml.overfill_never_worked"
prior_work_model = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name}/1")
never_work_model = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name2}/1")


# COMMAND ----------

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
df_prior_work_test = df[df['ever_worked']==1]
df_never_work_test = df[df['ever_worked']==0]

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

df2 = pd.concat([df_prior_work_test, df_never_work_test])

# COMMAND ----------

now = datetime.now()
df2['as_of_date'] = now
pred_df = spark.createDataFrame(df2.reset_index())

# COMMAND ----------

# %sql
# drop table if exists bluecrew.ml.individual_overfill_prediction_output

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def cast_columns_to_match_schema(df: DataFrame, target_schema: DataFrame) -> DataFrame:
  """
  Cast columns in `df` to match the schema of `target_schema` if types are different.

  :param df: The DataFrame whose schema needs to be adjusted.
  :param target_schema: The DataFrame whose schema we want to match.
  :return: A new DataFrame with columns cast to match the target schema.
  """
  selectedColumns = [col for col in target_schema.columns if col in target_schema.schema.names]
  df = df.select(*selectedColumns)

  for field in target_schema.schema.fields:
    col_name = field.name
    target_type = field.dataType

    
    if col_name in df.columns:
      current_type = df.schema[col_name].dataType
      print(col_name, current_type, "target", target_type)
      if current_type != target_type:
        df = df.withColumn(col_name, col(col_name).cast(target_type))
        print(col_name)
  return df


# Example usage:
# df1 = spark.read.parquet("path_to_first_schema.parquet")
# df2 = spark.read.parquet("path_to_second_schema.parquet")
# df1_casted = cast_columns_to_match_schema(df1, df2)
# df1_casted.printSchema()


# COMMAND ----------

initialDeltaTable = spark.read.format("delta").table('bluecrew.ml.individual_overfill_prediction_output')

updatedPredictions = cast_columns_to_match_schema(pred_df, initialDeltaTable)
updatedPredictions.printSchema()

# COMMAND ----------

# deltaTableSchema = spark.read.format("delta").table('bluecrew.ml.individual_overfill_prediction_output').schema

# # Select the columns from the new_predictions DataFrame that match the Delta table schema
# selectedColumns = [col for col in pred_df.columns if col in deltaTableSchema.names]
# updatedPredictions = pred_df.select(*selectedColumns)
# updatedPredictions = updatedPredictions.withColumn('SCHEDULE_ID', col('SCHEDULE_ID').cast('int'))
# updatedPredictions = updatedPredictions.withColumn('ever_worked', col('ever_worked').cast('float'))
# updatedPredictions = updatedPredictions.withColumn('prediction', col('prediction').cast('float'))
# updatedPredictions = updatedPredictions.withColumn('JOB_TYPE_TITLE_COUNT', col('JOB_TYPE_TITLE_COUNT').cast('double'))
# updatedPredictions = updatedPredictions.withColumn('TOTAL_JOB_COUNT', col('TOTAL_JOB_COUNT').cast('double'))

# deltaTable = spark.read.format("delta").table('bluecrew.ml.individual_overfill_prediction_output')
# deltaTable.printSchema()
# # Print the schema of the updatedPredictions DataFrame
# updatedPredictions.printSchema()

# COMMAND ----------

write_spark_table_to_databricks_schema(updatedPredictions, 'individual_overfill_prediction_output', 'bluecrew.ml', mode = 'append')

# COMMAND ----------

# MAGIC %md
# MAGIC # Need to add the post processing outputs to get to new predictions for Streamlit

# COMMAND ----------

# sdf2 = spark.createDataFrame(df2.reset_index())
# sdf2.createOrReplaceTempView('data')

# COMMAND ----------

updatedPredictions.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL_INDIVIDUAL_PREDICTIONS').save()


# COMMAND ----------

plt.figure(figsize=(15, 10))

sns.histplot(data=df2, x='predicted_prob', hue="JOB_TYPE",palette="tab20",
   alpha=.5, linewidth=0,bins=10, multiple='stack')


# sns.histplot(data=df2, x='predicted_prob', hue="JOB_TYPE",palette="Spectral",
#    alpha=.5, linewidth=0,bins=10, multiple='stack')
plt.show()

# COMMAND ----------

display(updatedPredictions)
updatedPredictions.createOrReplaceTempView('predictions')


# COMMAND ----------

account_managers = {'Brandi Watterson': 'brandi.watterson@bluecrewjobs.com',
 'Ashley Hovorka':'ahovorka@bluecrewjobs.com',
 'Brianna Poncsak':'brianna.poncsak@bluecrewjobs.com',
 'Celina Sanchez': 'celina.sanchez@bluecrewjobs.com',
 'Kacey Spence': 'kacey@bluecrewjobs.com',
 'Moira Mcasey': 'moira.mcasey@bluecrewjobs.com',
 'Molly Rez': 'molly@bluecrewjobs.com',
 'Nick Jameson': 'njameson@bluecrewjobs.com',
 'Ryan Galligan': 'ryan.galligan@bluecrewjobs.com',
 'Rg Galligan': 'ryan.galligan@bluecrewjobs.com',
 'Jessica Gagne': 'jessica.gagne@bluecrewjobs.com',
 'Nick Patrick': 'npatrick@bluecrewjobs.com',
 'Tyler McDanel': 'tyler.mcdanel@employbridge.com',
 'Quintin Gabler': 'quintin@bluecrewjobs.com'}

# COMMAND ----------

new_df = spark.sql("""
with expected_show_ups as (
  select p.job_id, 
  sum(case when app.user_id is not null then predicted_prob else 0 end) as expected_show_ups, 
  count(case when app.user_id is not null then predicted_prob else null end) as current_sign_ups
  from predictions p
  left join bc_foreign_snowflake.dm.dm_cm_job_application_status app
  on p.user_id = app.user_id and p.job_id = app.job_id and app.USER_CURRENT_JOB_APPLICATION_STATUS = 'SUCCESS'
  
  group by 1)
select esu.*, j.JOB_NEEDED_LAST_COUNT, j.job_overfill, j.COMPANY_ACCOUNT_MANAGER, j.JOB_START_DATE_TIME, j.COMPANY_NAME,
timediff(HOUR, current_timestamp(),j.JOB_START_DATE_TIME ) as time_until_job_start,
expected_show_ups/j.JOB_NEEDED_LAST_COUNT*100 as current_expected_work_rate,
esu.current_sign_ups/j.JOB_NEEDED_LAST_COUNT*100 as current_fill_rate
from expected_show_ups esu
left join bc_foreign_snowflake.dm.fact_job j
on esu.job_id = j.job_id
where timediff(HOUR, current_timestamp(),j.JOB_START_DATE_TIME ) between 6 and 48
and job_needed_last_count > 0
and j.job_status != 'Cancelled'
and company_name not like 'WestRock - Bolingbrook, IL'
and expected_show_ups/j.JOB_NEEDED_LAST_COUNT*100 < 80
""")

display(new_df)




# COMMAND ----------

new_df_pandas = new_df.toPandas()
a = set(new_df_pandas['COMPANY_ACCOUNT_MANAGER'])
print(a)

for value in a:
  if value in account_managers:
    print(value)
  else:
    print('problem', value)

# COMMAND ----------

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



# COMMAND ----------

# set up the SMTP server
s = smtplib.SMTP(host='smtp.gmail.com', port=587)
 
# authenticate with the SMTP server
s.starttls()
s.login('daniel.streeter@bluecrewjobs.com', 'ctyh xqrr fluv bxjr')

for value in a:
  if value in account_managers:
    jobs_to_check = new_df_pandas.loc[new_df_pandas['COMPANY_ACCOUNT_MANAGER']==value, ['COMPANY_NAME','job_id']]
    job_text = []
    for company in set(jobs_to_check['COMPANY_NAME']):
      job_text.append(str(company)+ ": " + ", ".join(new_df_pandas.loc[new_df_pandas['COMPANY_NAME']==company, 'job_id']))
    # jobs_to_check = ", ".join(list(new_df_pandas.loc[new_df_pandas['COMPANY_ACCOUNT_MANAGER']==value,['COMPANY_NAME','job_id']]))
    job_string = "\n\n".join(job_text)

    # create the message
    msg = MIMEMultipart()
    msg['From'] = 'daniel.streeter@bluecrewjobs.com'
    msg['To'] = str(account_managers[value])
    # msg['To'] = 'ryan.bass@employbridge.com'
    # msg['To'] = 'jpenzotti@bluecrewjobs.com'
    # msg['To'] = 'daniel.streeter@employbridge.com'
    msg['Bcc'] = 'daniel.streeter@employbridge.com'
    msg['Subject'] = 'Shifts with Low Expected Work Rates'

    # add the message body
    message = f'Hello {value},\n \n \nBased on an expected work rate of less than 80%, we recommend using recommendations or applicant/eligible tracking in the Job Fill App to assist with the following jobs starting in the next 2 days: \n\n{job_string} \n \n \nA link to the app is: https://app.snowflake.com/vt51526/vha09841/#/streamlit-apps/STREAMLIT_APPS.PERSONALIZATION.H64CBIYTZHHAMWAJ?ref=snowsight_shared) \n \n \nPlease let us know if you have any questions or think this message was sent in error. \n \nThanks, \n \nDan Streeter'
    msg.attach(MIMEText(message, 'plain'))

    # if value in ['Nick Jameson', 'Brandi Watterson']:
    # send the message
    s.send_message(msg)


# log out of the SMTP server
s.quit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leadership email

# COMMAND ----------

# recipients = ['npatrick@bluecrewjobs.com']
recipients = ['mlaurinas@bluecrewjobs.com','jessica.gagne@bluecrewjobs.com', 'npatrick@bluecrewjobs.com', 'quintin@bluecrewjobs.com']

# COMMAND ----------

# set up the SMTP server
s = smtplib.SMTP(host='smtp.gmail.com', port=587)
 
# authenticate with the SMTP server
s.starttls()
s.login('daniel.streeter@bluecrewjobs.com', 'ctyh xqrr fluv bxjr')

message = ['Hello, \n \n \nBased on the currently successful applicants and an expected work rate of less than 80%, we have identified the following jobs starting in the next 2 days that may be at risk of having low work rates.']

for value in a:
  if value in account_managers:
    jobs_to_check = new_df_pandas.loc[new_df_pandas['COMPANY_ACCOUNT_MANAGER']==value, ['COMPANY_NAME','job_id']]
    job_text = []
    for company in set(jobs_to_check['COMPANY_NAME']):
      job_text.append(str(company)+ ": " + ", ".join(new_df_pandas.loc[new_df_pandas['COMPANY_NAME']==company, 'job_id']))
    # jobs_to_check = ", ".join(list(new_df_pandas.loc[new_df_pandas['COMPANY_ACCOUNT_MANAGER']==value,['COMPANY_NAME','job_id']]))
    job_string = "\n\t".join(job_text)

    

    # add the message body
    message.append(f'\n\n\n{value}:\n\t{job_string}')
            
            
message.append('\n \n \nA link to the app is: https://app.snowflake.com/vt51526/vha09841/#/streamlit-apps/STREAMLIT_APPS.PERSONALIZATION.H64CBIYTZHHAMWAJ?ref=snowsight_shared) \n \n \nPlease let us know if you have any questions or think this message was sent in error. \n \nThanks, \n \nDan Streeter')

final_message = "".join(message)
print(final_message)
    

    
    
# create the message
msg = MIMEMultipart()
msg['From'] = 'daniel.streeter@bluecrewjobs.com'

msg['To'] = ", ".join(recipients)
# msg['To'] = 'ryan.bass@employbridge.com'
# msg['To'] = 'jpenzotti@bluecrewjobs.com'
# msg['To'] = 'daniel.streeter@employbridge.com'
msg['Bcc'] = 'daniel.streeter@employbridge.com'
msg['Subject'] = 'Summary of Shifts with Low Expected Work Rates'

#add the message text
msg.attach(MIMEText(final_message, 'plain'))

# send the message
s.send_message(msg)


# log out of the SMTP server
s.quit()

# COMMAND ----------

