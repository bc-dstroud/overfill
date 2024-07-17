# Databricks notebook source
# MAGIC %md
# MAGIC # Individual Level Model Description
# MAGIC This notebook runs util notebooks, queries the foreign catalog to obtain individual level application event data, cleans it, and develops a model to predict individual-level show up rates.  It answers the question, "if someone applied, what was the outcome of that application?" by first checking to see if they worked the first shift of the applied to job.  If not, it checks when their latest cancellation occured and categorizes them into early cancellation, SNC, and NCNS.  This model will supplement the job-level overfill prediction model and apply its predictions to the applicant population.

# COMMAND ----------

# MAGIC %md
# MAGIC # Library installs and util runs

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

# start_date = datetime(2024, 2, 1)
start_date = datetime.today()-timedelta(days = 60)
now = datetime.now()
experiment_start_date = start_date + timedelta(days = 30)
end_date = (experiment_start_date + timedelta(days=30)).strftime("%Y-%m-%d")

sdf = jobs_query(start_date,end_date)


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


# COMMAND ----------

df = sdf.toPandas()

# COMMAND ----------

df = df[(df['target_var']=='Worked')|(df['target_var']=='No Cancel')]
df['Work'] = df['target_var'].apply(lambda x: 1 if x == "Worked" else 0)
df

# COMMAND ----------

df2 = df[['JOB_ID','Work']].groupby('JOB_ID').agg(workers = ('Work', 'sum'), applicants = ('JOB_ID','count'))
df2['first_shift_show_up_rate']= df2['workers']/df2['applicants']
# df2 = df2[df2['applicants']>3]
df2
# display(job_modeling)

# COMMAND ----------

sdf2 = spark.createDataFrame(df2.reset_index())

# COMMAND ----------

write_spark_table_to_databricks_schema(optimize_spark(sdf2), 'first_shift_show_up_rate', 'bluecrew.product_analytics')

# COMMAND ----------

df3 = spark.sql("""
  select * from bluecrew.ml.individual_overfill_prediction_output
                """)
display(df3)

# COMMAND ----------

df3.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL_INDIVIDUAL_PREDICTION_TRACKING').save()

# COMMAND ----------

df4 = spark.sql("""
  select * from bluecrew.ml.overfill_inference
                """)
display(df4)

# COMMAND ----------

df4.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL_JOB_PREDICTION_TRACKING').save()

# COMMAND ----------

sdf2.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'FIRST_SHIFT_SHOW_UP_RATE').save()
