# Databricks notebook source
# MAGIC %pip install scikit-plot
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/ml-modeling-utils

# COMMAND ----------

# MAGIC %run ./_resources/mlflow-utils

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
from xgboost import XGBClassifier, XGBRegressor

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking import MlflowClient


# COMMAND ----------

# MAGIC %md
# MAGIC # Job Level Predictions

# COMMAND ----------

query = """

SELECT * 
FROM PERSONALIZATION.OVERFILL_EVALUATION
"""
sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string'))

display(sdf)

# COMMAND ----------

df = optimize_spark(sdf).toPandas()
df['PREDICTED_SHOW_UP_RATE'] = df['MODELED_EXPECTED']/df['MODELED_APPLICANTS']
df['OBSERVED_SHOW_UP_RATE'] = df['MODELED_WORKED']/df['MODELED_APPLICANTS']
df.sort_values(by = 'WEIGHTED_ERROR', ascending = False).head(20)

# COMMAND ----------

sns.scatterplot(data=df, x='WORK_DELTA',y='SHOW_UP_PERCENTAGE_DELTA', hue = "WEIGHTED_ERROR",
   alpha=1)
plt.show()

# COMMAND ----------

sns.scatterplot(data=df[(df['MODELED_APPLICANTS']>5)&(df['MODELED_APPLICANTS']<50)], x='PREDICTED_SHOW_UP_RATE',y='OBSERVED_SHOW_UP_RATE', hue = "MODELED_APPLICANTS",
   alpha=.5)
plt.show()

# COMMAND ----------

sns.scatterplot(data=df, x='MODELED_EXPECTED',y='MODELED_WORKED', hue = "WEIGHTED_ERROR",
   alpha=.5)
plt.show()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import scipy

# Calculate the correlation between the two variables
r, p = scipy.stats.pearsonr(df['MODELED_EXPECTED'], df['MODELED_WORKED'])
plt.figure(figsize=(8,8))
# Add a trend line using sns.regplot
sns.regplot(data=df, x='MODELED_EXPECTED', y='MODELED_WORKED', scatter=False, color='red', label=f'Correlation: {r:.2f}')
sns.scatterplot(data=df, x='MODELED_EXPECTED',y='MODELED_WORKED', alpha=.5)

# Add a legend
plt.legend()
plt.title('Predicted vs. Observed Show Ups')
plt.xlabel('Model Prediction')
plt.ylabel('Observed Workers')
# Show the plot
plt.show()

# COMMAND ----------

(rmse, mae, r2, mape) = eval_metrics(df['MODELED_WORKED'], df['MODELED_EXPECTED'])

# Print out model metrics
print('Worked prediction metrics')
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

df2=df[df['MODELED_APPLICANTS']>0]
(rmse, mae, r2, mape) = eval_metrics(df2['OBSERVED_SHOW_UP_RATE'], df2['PREDICTED_SHOW_UP_RATE'],)

# Print out model metrics
print('Show Up Rate Metrics')
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)


# COMMAND ----------

# MAGIC %md
# MAGIC # Individual Level Predictions

# COMMAND ----------

query = """

SELECT * 
FROM PERSONALIZATION.OVERFILL_EVALUATION_INDIVIDUAL
"""
sdf2 = spark.read.format("snowflake").options(**options).option("query", query).load()

# sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string'))

sdf2

# COMMAND ----------

df2 = optimize_spark(sdf2).toPandas()
# df['PREDICTED_SHOW_UP_RATE'] = df['MODELED_EXPECTED']/df['MODELED_APPLICANTS']
# df['OBSERVED_SHOW_UP_RATE'] = df['MODELED_WORKED']/df['MODELED_APPLICANTS']
df2

# COMMAND ----------

# Calculate the correlation between the two variables
r, p = scipy.stats.pearsonr(df2['PREDICTED_PROB'], df2['WORKED'])

# Add a trend line using sns.regplot
sns.regplot(data=df2, x='PREDICTED_PROB', y='WORKED', scatter=False, color='red', label=f'Correlation: {r:.2f}')
sns.scatterplot(data=df2, x='PREDICTED_PROB',y='WORKED', alpha=.5)

# Add a legend
plt.legend()

# Show the plot
plt.show()

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

df2['work_prediction']=df2['PREDICTED_PROB'].apply(lambda x: 1 if x>=.7 else 0)


print(confusion_matrix(df2['WORKED'], df2['work_prediction']))


# COMMAND ----------

(rmse, mae, r2, mape) = eval_metrics(df2['WORKED'], df2['PREDICTED_PROB'])

# Print out model metrics
print('Worked prediction metrics')
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

df2['average_worked']=df2['WORKED'].mean()
(rmse, mae, r2, mape) = eval_metrics( df2['WORKED'], df2['average_worked'],)
# Print out model metrics
print('Worked prediction metrics')
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

# df2=df[df['MODELED_APPLICANTS']>0]
# (rmse, mae, r2, mape) = eval_metrics(df2['PREDICTED_SHOW_UP_RATE'], df2['OBSERVED_SHOW_UP_RATE'])

# # Print out model metrics
# print('Show Up Rate Metrics')
# print("  RMSE: %s" % rmse)
# print("  MAE: %s" % mae)
# print("  R2: %s" % r2)
# print("  MAPE: %s" % mape)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
y = df2['WORKED']
pred = df2['PREDICTED_PROB']
fpr, tpr, thresholds = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='Show Up Prediction')
display.plot()

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

y = df2['WORKED']
pred = df2['PREDICTED_PROB']
# y_pred = clf.decision_function(X_test)
RocCurveDisplay.from_predictions(y, pred)
plt.plot(np.arange(0,1,.01),np.arange(0,1,.01), linestyle = '--', c = 'r', label = 'Random')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# COMMAND ----------

import scikitplot as skplt
y_true = df2['WORKED']
y_prob = [1-df2['PREDICTED_PROB'], df2['PREDICTED_PROB']]


import numpy as np

def cumulative_classification_lift(y_true, y_prob):
    """
    Compute the cumulative lift for classification.
    
    :param y_true: Array-like, true binary labels (0 or 1).
    :param y_prob: Array-like, predicted probabilities for the positive class.
    :return: List of cumulative lift values.
    """
    # Sort y_true based on predicted probabilities in descending order
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n = len(y_true)
    cumulative_positive = np.cumsum(y_true_sorted)
    
    # Calculate cumulative percentage of actual positives
    cumulative_positive_percentage = cumulative_positive / np.arange(1, n + 1)
    
    # Overall percentage of actual positives in the dataset
    overall_positive_percentage = np.sum(y_true) / n
    
    cumulative_lifts = cumulative_positive_percentage / overall_positive_percentage
    
    return cumulative_lifts

cumulative_lift_values = cumulative_classification_lift(df2['WORKED'],df2['PREDICTED_PROB'])

# COMMAND ----------

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(y_true) + 1) / len(y_true), cumulative_lift_values, linestyle='-', color='b')
plt.axhline(y=1, color='red', linestyle='--')
plt.xlabel('Proportion of Samples')
plt.ylabel('Cumulative Lift')
plt.title('Cumulative Lift Curve')
plt.legend(['Model', 'Random'])
plt.grid(True)
plt.show()

# COMMAND ----------

df2['prob_decile']=(10*df2['PREDICTED_PROB']).apply(np.floor)/10

df2[['WORKED','prob_decile']].groupby('prob_decile').agg(['mean','count', 'sum'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Overfill Metric Graphs

# COMMAND ----------

headers = ['Variable Name', 'n', 'Point Estimate', 'std dev', 'Confidence Interval Lower', 'Confidence Interval Upper' ]

treatment_list = [['Average Talent Needed', 37, 9.76, 15.72, 4.69466992233717, 14.8253300776628],
['Average Talent Expected', 37, 11.81, 19.32, 5.58467066791057, 18.0353293320894],
['Average Talent Worked', 37, 7.95, 12.92, 3.78689156466898, 12.113108435331],
['Average Job-Level Work Rate', 37, 0.97, 0.55, 0.792777891684825, 1.14722210831518],
['Average Job-Level Fill Rate', 37, 1.46, 0.68, 1.24088902971942, 1.67911097028058],
['Average Job-Level Show Up Rate', 37, 0.71, 0.25, 0.629444496220375, 0.790555503779625],
['Average Job Overfill Requested', 37, 8.89, 14.99, 4.05989199337368, 13.7201080066263],
['Average Job-Level Overfill Sent Home Rate', 37, 0.01, 0.04, 0, 0.02288888060474]]

treatment_list_percents = [
['Average Job-Level Work Rate', 37, 0.97, 0.55, 0.792777891684825, 1.14722210831518],
['Average Job-Level Fill Rate', 37, 1.46, 0.68, 1.24088902971942, 1.67911097028058],
['Average Job-Level Show Up Rate', 37, 0.71, 0.25, 0.629444496220375, 0.790555503779625],
['Average Job-Level Overfill Sent Home Rate', 37, 0.01, 0.04, 0, 0.02288888060474]]

control_list = [
['Average Talent Needed', 4642, 1.86, 4.27, 1.73716238424288, 1.92016536281982],
['Average Talent Expected', 4642, 1.72, 3.84, 1.60953244859313, 1.77410655579112],
['Average Talent Worked', 4642, 1.36, 2.79, 1.27973841968094, 1.39931179444199],
['Average Job-Level Work Rate', 4642, 0.8, 0.58, 0.78331479692292, 0.808172344364284],
['Average Job-Level Fill Rate', 4642, 0.96, 1.17, 0.926341917930718, 0.976485591217607],
['Average Job-Level Show Up Rate', 4642, 0.87, 0.29, 0.86165739846146, 0.874086172182142],
['Average Job Overfill Requested', 4642, 0.53, 2.65, 0.453765882492652, 0.567339159595436],
['Average Job-Level Overfill Sent Home Rate', 4642, 0.01, 0.08, 0.00769859267902347, 0.011127219912315]]

control_list_percents = [
['Average Job-Level Work Rate', 4642, 0.8, 0.58, 0.78331479692292, 0.808172344364284],
['Average Job-Level Fill Rate', 4642, 0.96, 1.17, 0.926341917930718, 0.976485591217607],
['Average Job-Level Show Up Rate', 4642, 0.87, 0.29, 0.86165739846146, 0.874086172182142],
['Average Job-Level Overfill Sent Home Rate', 4642, 0.01, 0.08, 0.00769859267902347, 0.011127219912315]]

# COMMAND ----------

treatment_df = pd.DataFrame(treatment_list_percents, columns = headers)
treatment_df['errors']=treatment_df['std dev']/np.sqrt(treatment_df['n'])
treatment_df['group'] = 'Treatment'

control_df = pd.DataFrame(control_list_percents, columns = headers)
control_df['errors']=control_df['std dev']/np.sqrt(control_df['n'])
control_df['group'] = 'Unfiltered Control'

final_df = pd.concat([treatment_df, control_df]).reset_index()
final_df

# COMMAND ----------


query = """

SELECT * 
FROM PERSONALIZATION.OVERFILL_EVALUATION_ALL_STATS
"""
sdf3 = spark.read.format("snowflake").options(**options).option("query", query).load()

# sdf = sdf.withColumn("JOB_ID", sdf["JOB_ID"].cast('string'))

sdf3


# COMMAND ----------

experiment_df = optimize_spark(sdf3).toPandas()

# COMMAND ----------

experiment_df

# COMMAND ----------

import statsmodels.formula.api as smf
def ab_test(df = df, y_var = 'total_spend', covariates = 'grp_code', grp_codes_to_test= ['c0', 'v1'], verbose = False):
    """Run an AB test between two groups to identify confidence intervals for the difference in the mean of the response variable

    Keyword arguments:
    df -- the pandas dataframe containing experiment data (default df)
    y_var -- the response variable (default total_spend)
    covariates -- string of independent variables to consider when building the regression model.  It should be in the form 'x1 + x2 + ... + xn' (default 'grp_code')
    grp_codes_to_test -- list of experiment group codes to test (default ['c0', 'v1'])
    verbose -- whether or not to print the full ols summary (default False)

    """
    # Builds formula used in the statsmodels api
    formula = f'{y_var} ~ {covariates}'

    # Calls and fits statsmodels api for an ols regression model for a subset of the df
    mod = smf.ols(formula = formula, data = df[df['experiment_group'].isin(grp_codes_to_test)])
    res = mod.fit()

    # Outputs the full statsmodel summary
    if verbose:
        print(f'Testing groups {grp_codes_to_test[0]} against {grp_codes_to_test[1]}:')
        print(res.summary())
    
    # Collects the relevant parameters from the model
    mean_diff = res.params.loc[f'experiment_group[T.{grp_codes_to_test[1]}]']
    conf_int = res.conf_int().loc[f'experiment_group[T.{grp_codes_to_test[1]}]']
    p_value = res.pvalues.loc[f'experiment_group[T.{grp_codes_to_test[1]}]']    
    std_err = res.bse.loc[f'experiment_group[T.{grp_codes_to_test[1]}]']    
    return [mean_diff, std_err, conf_int[0],conf_int[1], p_value]


# COMMAND ----------

print(experiment_df.describe())

# COMMAND ----------

# covariates = 'TYPE'
# y_var = 'WORK_RATE'
# formula = f'{y_var} ~ {covariates}'

# # Calls and fits statsmodels api for an ols regression model for a subset of the df
# mod = smf.ols(formula = formula, data = experiment_df)
# res = mod.fit()

# COMMAND ----------

# ab_test(df = experiment_df, covariates = 'TYPE', grp_codes_to_test=['Control', 'Treatment'], y_var = 'WORK_RATE', verbose=True)

# COMMAND ----------

# # Defines a list of response variables and groups to use for the abtest() function
# response_variables = ['WORK_RATE',	'FILL_RATE',	'SHOW_UP_RATE',	'OVERFILL_SENT_HOME_RATE'	]
# variant_pairs = [['Control','Treatment']]

# # tracker for the abtest results
# result_list = []

# # runs an ols regression for each variable pair
# for var in response_variables:
#     for variant_pair in variant_pairs:
#         result_list.append([variant_pair, var]+ab_test(df = experiment_df, covariates = 'TYPE', grp_codes_to_test=variant_pair, y_var = var, verbose=False))

# COMMAND ----------

# result_df = pd.DataFrame(result_list, columns=['var_pair', 'response_var', 'mean_diff', 'std_error', 'conf_int_lower', 'conf_int_upper', 'p_value'])
# result_df['bonferroni_adjusted_p'] = result_df['p_value']*result_df.shape[0]
# result_df['conclusion'] = result_df['bonferroni_adjusted_p'].apply(lambda x: 'Reject the null hypothesis' if x < .05 else 'Fail to reject the null hypothesis')
# result_df

# COMMAND ----------

table_name = "bluecrew.ml.job_fill_experiment_results"


propensity_result_sdf =spark.read.table(table_name)
propensity_result_df = optimize_spark(propensity_result_sdf).toPandas()

# COMMAND ----------

propensity_result_df

# COMMAND ----------

filtered_control_ids = set(propensity_result_df.loc[propensity_result_df['EXPERIMENT_GROUP']=='control','JOB_ID'])
filtered_control_ids

# COMMAND ----------

experiment_df['experiment_group'] = experiment_df.apply(lambda x: 'Filtered Control' if x['JOB_ID'] in filtered_control_ids else x['TYPE'], axis = 1 )


experiment_df.loc[experiment_df['experiment_group']!='Control','experiment_group']

# COMMAND ----------

# Defines a list of response variables and groups to use for the abtest() function
response_variables = ['WORK_RATE',	'FILL_RATE',	'SHOW_UP_RATE',	'OVERFILL_SENT_HOME_RATE'	]
variant_pairs = [['Filtered Control','Treatment']]

# tracker for the abtest results
result_list = []

# runs an ols regression for each variable pair
for var in response_variables:
    for variant_pair in variant_pairs:
        result_list.append([variant_pair, var]+ab_test(df = experiment_df, covariates = 'experiment_group', grp_codes_to_test=variant_pair, y_var = var, verbose=True))

# COMMAND ----------

new_df = experiment_df[experiment_df['experiment_group']=='Filtered Control'].describe().T
new_df

# COMMAND ----------

over_100_df = experiment_df[experiment_df['WORK_RATE']>1].groupby('experiment_group').count()
over_100_df

# COMMAND ----------

new_df['std_error']= new_df['std']/np.sqrt(new_df['count'])
new_df['2std_error']= 2*new_df['std_error']
new_df[['count', 'mean', '2std_error']]
new_df.loc['SHIFT_NEEDED']