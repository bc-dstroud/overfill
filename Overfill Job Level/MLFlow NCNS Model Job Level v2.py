# Databricks notebook source
# MAGIC %md
# MAGIC # Job Level Model
# MAGIC This notebook loads, cleans, and develops a model to predict job-level work rates (1-NCNS rate) that will help with setting an overfill rate.

# COMMAND ----------

# MAGIC %md
# MAGIC #Preprocessing

# COMMAND ----------

#Imported libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import to_date, current_timestamp
from math import exp 
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK 
import mlflow
from mlflow.models.signature import infer_signature
# import shap
import databricks.automl as db_automl

# COMMAND ----------

# This builds the FeatureImportance class, which is used to easily extract feature importances from the model pipeline
# I know some of these are duplicates, but these are the class dependencies.  Should we turn this class into something that can be used by all of BC Data Science?
import numpy as np  
import pandas as pd  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import plotly.express as px


class FeatureImportance:

    """
    
    Extract & Plot the Feature Names & Importance Values from a Scikit-Learn Pipeline.
    
    The input is a Pipeline that starts with a ColumnTransformer & ends with a regression or classification model. 
    As intermediate steps, the Pipeline can have any number or no instances from sklearn.feature_selection.

    Note: 
    If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns, 
    it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns 
    to the dataset that didn't exist before, so there should come last in the Pipeline.
    
    
    Parameters
    ----------
    pipeline : a Scikit-learn Pipeline class where the a ColumnTransformer is the first element and model estimator is the last element
    verbose : a boolean. Whether to print all of the diagnostics. Default is False.
    
    Attributes
    __________
    column_transformer_features :  A list of the feature names created by the ColumnTransformer prior to any selectors being applied
    transformer_list : A list of the transformer names that correspond with the `column_transformer_features` attribute
    discarded_features : A list of the features names that were not selected by a sklearn.feature_selection instance.
    discarding_selectors : A list of the selector names corresponding with the `discarded_features` attribute
    feature_importance :  A Pandas Series containing the feature importance values and feature names as the index.    
    plot_importances_df : A Pandas DataFrame containing the subset of features and values that are actually displaced in the plot. 
    feature_info_df : A Pandas DataFrame that aggregates the other attributes. The index is column_transformer_features. The transformer column contains the transformer_list.
        value contains the feature_importance values. discarding_selector contains discarding_selectors & is_retained is a Boolean indicating whether the feature was retained.
    
    
    
    """
    def __init__(self, pipeline, verbose=False):
        self.pipeline = pipeline
        self.verbose = verbose


    def get_feature_names(self, verbose=None):  

        """

        Get the column names from the a ColumnTransformer containing transformers & pipelines

        Parameters
        ----------
        verbose : a boolean indicating whether to print summaries. 
            default = False


        Returns
        -------
        a list of the correct feature names

        Note: 
        If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns, 
        it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns 
        to the dataset that didn't exist before, so there should come last in the Pipeline.

        Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525 

        """

        if verbose is None:
            verbose = self.verbose
            
        if verbose: print('''\n\n---------\nRunning get_feature_names\n---------\n''')
        
        column_transformer = self.pipeline[0]        
        assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
        check_is_fitted(column_transformer)

        new_feature_names, transformer_list = [], []

        for i, transformer_item in enumerate(column_transformer.transformers_): 
            
            transformer_name, transformer, orig_feature_names = transformer_item
            orig_feature_names = list(orig_feature_names)
            
            if verbose: 
                print('\n\n', i, '. Transformer/Pipeline: ', transformer_name, ',', 
                      transformer.__class__.__name__, '\n')
                print('\tn_orig_feature_names:', len(orig_feature_names))

            if transformer == 'drop':
                    
                continue
                
            if isinstance(transformer, Pipeline):
                # if pipeline, get the last transformer in the Pipeline
                transformer = transformer.steps[-1][1]

            if hasattr(transformer, 'get_feature_names'):

                if 'input_features' in transformer.get_feature_names.__code__.co_varnames:

                    names = list(transformer.get_feature_names(orig_feature_names))

                else:

                    names = list(transformer.get_feature_names())

            elif hasattr(transformer,'indicator_') and transformer.add_indicator:
                # is this transformer one of the imputers & did it call the MissingIndicator?

                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [orig_feature_names[idx] + '_missing_flag'\
                                      for idx in missing_indicator_indices]
                names = orig_feature_names + missing_indicators

            elif hasattr(transformer,'features_'):
                # is this a MissingIndicator class? 
                missing_indicator_indices = transformer.features_
                missing_indicators = [orig_feature_names[idx] + '_missing_flag'\
                                      for idx in missing_indicator_indices]

            else:

                names = orig_feature_names

            if verbose: 
                print('\tn_new_features:', len(names))
                print('\tnew_features:\n', names)

            new_feature_names.extend(names)
            transformer_list.extend([transformer_name] * len(names))
        
        self.transformer_list, self.column_transformer_features = transformer_list,\
                                                                    new_feature_names

        return new_feature_names

    
    def get_selected_features(self, verbose=None):
        """

        Get the Feature Names that were retained after Feature Selection (sklearn.feature_selection)

        Parameters
        ----------
        verbose : a boolean indicating whether to print summaries. default = False

        Returns
        -------
        a list of the selected feature names


        """

        if verbose is None:
            verbose = self.verbose

        assert isinstance(self.pipeline, Pipeline), "Input isn't a Pipeline"

        features = self.get_feature_names()
        
        if verbose: print('\n\n---------\nRunning get_selected_features\n---------\n')
            
        all_discarded_features, discarding_selectors = [], []

        for i, step_item in enumerate(self.pipeline.steps[:]):
            
            step_name, step = step_item

            if hasattr(step, 'get_support'):

                if verbose: print('\nStep ', i, ": ", step_name, ',', 
                                  step.__class__.__name__, '\n')
                    
                check_is_fitted(step)

                feature_mask_dict = dict(zip(features, step.get_support()))
                
                features = [feature for feature, is_retained in feature_mask_dict.items()\
                            if is_retained]
                                         
                discarded_features = [feature for feature, is_retained in feature_mask_dict.items()\
                                      if not is_retained]
                
                all_discarded_features.extend(discarded_features)
                discarding_selectors.extend([step_name] * len(discarded_features))
                
                
                if verbose: 
                    print(f'\t{len(features)} retained, {len(discarded_features)} discarded')
                    if len(discarded_features) > 0:
                        print('\n\tdiscarded_features:\n\n', discarded_features)

        self.discarded_features, self.discarding_selectors = all_discarded_features,\
                                                                discarding_selectors
        
        return features

    def get_feature_importance(self):
        
        """
        Creates a Pandas Series where values are the feature importance values from the model and feature names are set as the index. 
        
        This Series is stored in the `feature_importance` attribute.

        Returns
        -------
        A pandas Series containing the feature importance values and feature names as the index.
        
        """
        
        assert isinstance(self.pipeline, Pipeline), "Input isn't a Pipeline"

        features = self.get_selected_features()
             
        assert hasattr(self.pipeline[-1], 'feature_importances_'),\
            "The last element in the pipeline isn't an estimator with a feature_importances_ attribute"
        
        importance_values = self.pipeline[-1].feature_importances_
        
        assert len(features) == len(importance_values),\
            "The number of feature names & importance values doesn't match"
        
        feature_importance = pd.Series(importance_values, index=features)
        self.feature_importance = feature_importance
        
        # create feature_info_df
        column_transformer_df =\
            pd.DataFrame(dict(transformer=self.transformer_list),
                         index=self.column_transformer_features)

        discarded_features_df =\
            pd.DataFrame(dict(discarding_selector=self.discarding_selectors),
                         index=self.discarded_features)

        importance_df = self.feature_importance.rename('value').to_frame()

        self.feature_info_df = \
            column_transformer_df\
            .join([importance_df, discarded_features_df])\
            .assign(is_retained = lambda df: ~df.value.isna())        


        return feature_importance
        
    
    def plot(self, top_n_features=100, rank_features=True, max_scale=True, 
             display_imp_values=True, display_imp_value_decimals=1,
             height_per_feature=25, orientation='h', width=750, height=None, 
             str_pad_width=15, yaxes_tickfont_family='Courier New', 
             yaxes_tickfont_size=15):
        """

        Plot the Feature Names & Importances 


        Parameters
        ----------

        top_n_features : the number of features to plot, default is 100
        rank_features : whether to rank the features with integers, default is True
        max_scale : Should the importance values be scaled by the maximum value & mulitplied by 100?  Default is True.
        display_imp_values : Should the importance values be displayed? Default is True.
        display_imp_value_decimals : If display_imp_values is True, how many decimal places should be displayed. Default is 1.
        height_per_feature : if height is None, the plot height is calculated by top_n_features * height_per_feature. 
        This allows all the features enough space to be displayed
        orientation : the plot orientation, 'h' (default) or 'v'
        width :  the width of the plot, default is 500
        height : the height of the plot, the default is top_n_features * height_per_feature
        str_pad_width : When rank_features=True, this number of spaces to add between the rank integer and feature name. 
            This will enable the rank integers to line up with each other for easier reading. 
            Default is 15. If you have long feature names, you can increase this number to make the integers line up more.
            It can also be set to 0.
        yaxes_tickfont_family : the font for the feature names. Default is Courier New.
        yaxes_tickfont_size : the font size for the feature names. Default is 15.

        Returns
        -------
        plot

        """
        if height is None:
            height = top_n_features * height_per_feature
            
        # prep the data
        
        all_importances = self.get_feature_importance()
        n_all_importances = len(all_importances)
        
        plot_importances_df =\
            all_importances\
            .nlargest(top_n_features)\
            .sort_values()\
            .to_frame('value')\
            .rename_axis('feature')\
            .reset_index()
                
        if max_scale:
            plot_importances_df['value'] = \
                                plot_importances_df.value.abs() /\
                                plot_importances_df.value.abs().max() * 100
            
        self.plot_importances_df = plot_importances_df.copy()
        
        if len(all_importances) < top_n_features:
            title_text = 'All Feature Importances'
        else:
            title_text = f'Top {top_n_features} (of {n_all_importances}) Feature Importances'       
        
        if rank_features:
            padded_features = \
                plot_importances_df.feature\
                .str.pad(width=str_pad_width)\
                .values
            
            ranked_features =\
                plot_importances_df.index\
                .to_series()\
                .sort_values(ascending=False)\
                .add(1)\
                .astype(str)\
                .str.cat(padded_features, sep='. ')\
                .values

            plot_importances_df['feature'] = ranked_features
        
        if display_imp_values:
            text = plot_importances_df.value.round(display_imp_value_decimals)
        else:
            text = None

        # create the plot 
        
        fig = px.bar(plot_importances_df, 
                     x='value', 
                     y='feature',
                     orientation=orientation, 
                     width=width, 
                     height=height,
                     text=text)
        fig.update_layout(title_text=title_text, title_x=0.5) 
        fig.update(layout_showlegend=False)
        fig.update_yaxes(tickfont=dict(family=yaxes_tickfont_family, 
                                       size=yaxes_tickfont_size),
                         title='')
        fig.show()

# COMMAND ----------

# Setting up database access
sfUser = dbutils.secrets.get(scope="my_secrets", key="snowflake-user")
SFPassword = dbutils.secrets.get(scope="my_secrets", key="snowflake-password")
 
options = {
  "sfUrl": "vha09841.snowflakecomputing.com",
  "sfUser": sfUser,
  "SFPassword": SFPassword,
  "sfDataBase": "BLUECREW",
  "sfSchema": "PERSONALIZATION",
  "sfWarehouse": "COMPUTE_WH"
}
start_date = '2023-01-01'
end_date = '2023-12-01'

startdate = pd.to_datetime(start_date).date()
enddate = pd.to_datetime(start_date).date()

# COMMAND ----------

# Snowflake Query
query = f"""

select jacw.*, c.company_origin from BLUECREW.PERSONALIZATION.JOB_APPLICATION_CANCELLATION_WORKED jacw
inner join dm.dm_companies c
on jacw.company_id = c.company_id
where 1=1
-- and job_type = 'Event Staff' 
and START_TIME >= '{start_date}'
and START_TIME <=  '{end_date}'
"""

sdf = spark.read.format("snowflake").options(**options).option("query", query).load()
new_df = sdf.withColumn("JOB_ID",  sdf["JOB_ID"].cast('int')).withColumn("START_TIME",to_date("START_TIME"))




# COMMAND ----------

# For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html

fe = FeatureEngineeringClient()


model_feature_lookups = [
      #This is a basic feature lookup that doesn't have a timestamp key
      FeatureLookup(
          table_name='feature_store.dev.calendar',
          lookup_key="START_TIME"
      )
      ,
      #This is a feature lookup that demonstrates how to use point in time based lookups for training sets
      FeatureLookup(
        table_name='feature_store.dev.jobs_data',
        feature_names=['COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'WAGE_DELTA', 'JOB_NEEDED_LAST_COUNT','JOB_SHIFTS','SCHEDULE_NAME_UPDATED','POSTING_LEAD_TIME_DAYS',"ELIGIBLE_USERS", "ACTIVE_USERS_7_DAYS", "COUNTY", "ELIGIBLE_CMS_1_MILE", "ELIGIBLE_CMS_5_MILE","ELIGIBLE_CMS_10_MILE", 'ELIGIBLE_CMS_15_MILE', "ACTIVE_CMS_1_MILE", "ACTIVE_CMS_5_MILE", "ACTIVE_CMS_10_MILE", "ACTIVE_CMS_15_MILE", "JOB_TYPE_TITLE_COUNT", "TOTAL_JOB_COUNT", "TOTAL_CMS_REQUIRED"],
        lookup_key="JOB_ID",
        timestamp_lookup_key="JOB_CREATED_AT")
]
training_set = fe.create_training_set(
    df = new_df, # joining the original Dataset, with our FeatureLookupTable
    feature_lookups=model_feature_lookups,
    exclude_columns=[], # exclude id columns as we don't want them as feature
    label='SUCCESSFULLY_WORKED'
)

training_pd = training_set.load_df()
display(training_pd)

# COMMAND ----------

# Converts the Spark df + features to a pandas DataFram and turns boolean columns into integers
#Probably should do the following instead:
# for idx, (col, dtype) in enumerate(zip(sdf.columns, sdf.dtypes)):
#   if dtype[1].startswith('decimal'):
#     sdf = sdf.withColumn(col.lower(), F.col(col).cast('int'))

df = training_pd.toPandas()
df = df.astype({'JOB_WAGE':'float', 'day_of_week':'string', 'COUNTY_JOB_TYPE_TITLE_AVG_WAGE':'float', 'WAGE_DELTA':'float', 'JOB_NEEDED_LAST_COUNT':'float', 'JOB_OVERFILL':'float', 'JOB_SHIFTS':'float', 'INVITED_WORKER_COUNT':'float','POSTING_LEAD_TIME_DAYS':'float',"ELIGIBLE_USERS":'float', "ACTIVE_USERS_7_DAYS":'float', 'SUCCESSFULLY_WORKED':'float', 'TOTAL_SUCCESSFUL_SIGN_UPS':'float', "ELIGIBLE_CMS_1_MILE":'float', "ELIGIBLE_CMS_5_MILE":'float',"ELIGIBLE_CMS_10_MILE":'float', 'ELIGIBLE_CMS_15_MILE':'float', "ACTIVE_CMS_1_MILE":'float', "ACTIVE_CMS_5_MILE":'float', "ACTIVE_CMS_10_MILE":'float', "ACTIVE_CMS_15_MILE":'float', "JOB_TYPE_TITLE_COUNT":'float', "TOTAL_JOB_COUNT":'float', "TOTAL_CMS_REQUIRED":'float', 'JOB_NEEDED_ORIGINAL_COUNT':'float'})
bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))

# COMMAND ----------

# 
df['Work'] = df['SUCCESSFULLY_WORKED']/df['TOTAL_SUCCESSFUL_SIGN_UPS']
df['CM_Count_Ratio']= df['TOTAL_CMS_REQUIRED']/df['TOTAL_JOB_COUNT']
df2 = df[(df['Work'].isnull() == False) & (df['Work']<=1)&(df['NEEDED']>=5)&(df['POSTING_LEAD_TIME_DAYS']>0)&(df['JOB_NEEDED_LAST_COUNT']>=3)&(df['TOTAL_SUCCESSFUL_SIGN_UPS']>=3)&(df['COMPANY_ORIGIN']!="EB")]
# df2 = df[(df['WORKED_NCNS']!='Neither')&(df['CANCEL_CREATED_AT'].isnull())&(df['JOB_NEEDED_LAST_COUNT']>=5)&(df['JOB_SHIFTS'] == 1)]
# df2 = df[(df['WORKED_NCNS']!='Neither')&(df['CANCEL_CREATED_AT'].isnull())&(df['JOB_NEEDED_LAST_COUNT']>=3)]

df2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Turning the dataframe into a job-based df with NCNS rate as the predictor variable
# MAGIC

# COMMAND ----------

# Defines the columns that correspond to a job and creates final dataset to split for training and testing.
cols_to_drop = ['SUCCESSFULLY_WORKED', 'TOTAL_SUCCESSFUL_SIGN_UPS', 'JOB_STATUS_ENUM', 'JOB_STATUS', 'JOB_OVERFILL', 'INVITED_WORKER_COUNT', 'SEGMENT_INDEX', 'NEEDED', 'JOB_NEEDED_LAST_COUNT', 'SIGN_UP_JOIN_COUNT', 'START_TIME', 'POSITION_ID', 'COMPANY_ID', 'COUNTY_JOB_TYPE_TITLE_AVG_WAGE']
df4 = df2.drop(columns=cols_to_drop)
df5 = df4.set_index('JOB_ID')
df5

# COMMAND ----------

# Splits DF into train and test set
y = df5['Work']
X = df5.drop(columns=['Work'])
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)



# COMMAND ----------

# Checks for missing values and determines shape of X_train
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]

print("Columns with missing values :", cols_with_missing)
print("X_train_ful shape :", X_train_full.shape)

# COMMAND ----------

# IDs categorical and numeric columns for use in modeling and for splitting into proper pipeline.
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 30 and X_train_full[cname].dtype in ["object", "string"]]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int32', 'int64', 'float64','decimal']]

print('categorical columns :', categorical_cols)
print('numerical columns :', numerical_cols)

# COMMAND ----------

# IDs new columns subset
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

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

# Evaluate metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes from dstroud:
# MAGIC 1. **Add the Model to the Pipeline:**
# MAGIC    - Include your `RandomForestRegressor` as a step in the pipeline. Let's name this step 'model'.
# MAGIC
# MAGIC 2. **Adjust the `params_to_xgb` Function:**
# MAGIC    - This function will no longer be necessary, as the parameters will be passed directly to the model in the pipeline.
# MAGIC
# MAGIC 3. **Modify the `train_model` Function:**
# MAGIC    - Update this function to use the pipeline for training and prediction.
# MAGIC    - The parameter names need to be updated to reflect that they are part of the 'model' step in the pipeline.
# MAGIC
# MAGIC
# MAGIC This code integrates the `RandomForestRegressor` into the pipeline and uses Hyperopt for optimization, with the parameters correctly prefixed for use within the pipeline. Remember to define your preprocessor and evaluation metrics (like `eval_metrics`) as per your existing code or requirements.

# COMMAND ----------

# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from hyperopt import hp, fmin, tpe, STATUS_OK
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
#     booster = my_pipeline.named_steps['model']
#     shap_values = shap.TreeExplainer(booster).shap_values(X_train2, y=y_train)
#     shap.summary_plot(shap_values, X_train2, feature_names=display_cols, plot_size=(14,6), max_display=10, show=False)
#     plt.savefig("summary_plot.png", bbox_inches="tight") 
#     plt.close()
#     mlflow.log_artifact("summary_plot.png")

#     # Predict and evaluate
#     preds = my_pipeline.predict(X_valid)
#     (rmse, mae, r2) = eval_metrics(y_valid, preds)
#     mlflow.log_metric("rmse", rmse)
#     mlflow.log_metric("r2", r2)
#     mlflow.log_metric("mae", mae)
#     return {'status': STATUS_OK, 'loss': rmse}

# # Define your search space
# search_space = {
#     'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
#     'n_estimators': scope.int(hp.quniform('n_estimators', 25, 150, 25))
# }

# # Hyperopt optimization
# spark_trials = SparkTrials(parallelism=4)
# best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=50, trials=spark_trials, rstate=np.random.default_rng(seed=42))

# COMMAND ----------

#Would like to get help from David updating the pipeline portion.  Hyperopt code I have only works with xgb.train function having the parameters.  Is it as easy as adding 'model__' to the front of the parameters?
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

encoder = my_pipeline.fit(X_train)
X_train2 = encoder.transform(X_train)
X_valid2 = encoder.transform(X_valid)




def params_to_xgb(params):
  return {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': int(params['max_depth']),
    'learning_rate': exp(params['log_learning_rate']), # exp() here because hyperparams are in log space
    'reg_alpha': exp(params ['log_reg_alpha']),
    'reg_lambda': exp(params ['log_reg_lambda']),
    'gamma': exp(params['log_gamma']),
    'min_child_weight':exp(params ['log_min_child_weight']),
    'importance_type': 'total_gain',
    'seed': 0
  }
def train_model(params):
  train = xgb.DMatrix(data=X_train2, label=y_train, enable_categorical=True)
  test = xgb.DMatrix(data=X_valid2, label=y_valid, enable_categorical=True)
  booster = xgb.train(params=params_to_xgb(params), dtrain=train, num_boost_round=1000, evals=[(test, "test")], early_stopping_rounds=50)
  # shap_values = shap.TreeExplainer(booster).shap_values(X_train2, y=y_train)

  # shap.summary_plot(shap_values, X_train2, feature_names=display_cols, plot_size= (14,6), max_display=10, show=False)
  # plt.savefig("summary_plot.png", bbox_inches="tight") 
  # plt.close () 
  # mlflow.log_artifact ("summary_plot.png")

  preds = booster.predict(test)
  (rmse, mae, r2) = eval_metrics(y_valid, preds)
  # mlflow.log_param('n_estimators', n_estimators)
  # mlflow.log_param('max_depth', max_depth)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)
  return {'status': STATUS_OK, 'loss': booster.best_score, 'booster': booster.attributes()}

search_space = {
  'max_depth':hp.quniform('max_depth', 3, 30, 1),
  # use uniform over loguniform here simply to make metrics show up better in mlflow comparison, in logspace
  'log_learning_rate': hp.uniform('log_learning_rate', -3, 0),
  'log_reg_alpha':hp.uniform('log_reg_alpha', -5, -1),
  'log_reg_lambda':hp.uniform('log_reg_lambda', 1, 8),
  'log_gamma':hp.uniform('log_gamma', -6, -1),
  'log_min_child_weight': hp.uniform('log_min_child_weight', -1, 3)
}

spark_trials = SparkTrials(parallelism=4)
best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=50, trials=spark_trials, rstate=np.random.default_rng(seed=42))

# COMMAND ----------

#Would like to get help from David updating the pipeline portion.  Hyperopt code I have only works with xgb.train function having the parameters.  Is it as easy as adding 'model__' to the front of the parameters?
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor)]) # add model to pipeline ~ dstroud

encoder = my_pipeline.fit(X_train)
X_train2 = encoder.transform(X_train)
X_valid2 = encoder.transform(X_valid)
display_cols = encoder.get_feature_names_out(input_features=None)

def params_to_xgb(params): # function not necessary if implement code above ~ dstroud
  return {
    'criterion':'squared_error',
    'max_depth': int(params['max_depth']),
    'n_estimators': int(params['n_estimators']),
    'seed': 0
  }

def train_model(params): # update for pipeline training and prediction. ie, my_pipeline.set_params(**params).fit(X_train, y_train) ~ dstroud
  params['max_depth'] = int(params['max_depth'])
  params['n_estimators'] = int(params['n_estimators'])
  booster = RandomForestRegressor(**params)
  booster.fit(X_train2, y_train)
  shap_values = shap.TreeExplainer(booster).shap_values(X_train2, y=y_train)

  shap.summary_plot(shap_values, X_train2, feature_names=display_cols, plot_size= (14,6), max_display=15, show=False)
  plt.savefig("summary_plot.png", bbox_inches="tight") 
  plt.close () 
  mlflow.log_artifact ("summary_plot.png")

  preds = booster.predict(X_valid2)
  (rmse, mae, r2) = eval_metrics(y_valid, preds)
  # mlflow.log_param('n_estimators', n_estimators)
  # mlflow.log_param('max_depth', max_depth)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)
  return {'status': STATUS_OK, 'loss': -r2}

search_space = {
  'max_depth':hp.quniform('max_depth', 3, 15, 1),
  'n_estimators':hp.quniform('n_estimators', 25, 150, 25)
}

spark_trials = SparkTrials(parallelism=4)
best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=50, trials=spark_trials, rstate=np.random.default_rng(seed=42))

# COMMAND ----------

print(best_params)

# COMMAND ----------

max_depth= int(best_params['max_depth'])
n_estimators=int(best_params['n_estimators'])
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)


# Evaluate the model
#Training Performance:
print("Training Performance:")
(rmse, mae, r2) = eval_metrics(y_train, my_pipeline.predict(X_train))

# Print out model metrics
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

#Test Performance:
print("Test Performance:")
(rmse, mae, r2) = eval_metrics(y_valid, preds)

# Print out model metrics
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)


# COMMAND ----------

feature_importance = FeatureImportance(my_pipeline)
feature_importance.plot(top_n_features=25)

# COMMAND ----------

# Builds DataFrame with predicted and actual values
a = y_valid.reset_index()
a = a.rename(columns={'Work':'Actual Work Rate'})
a['Predicted Work Rate'] = preds
a['Delta'] = abs(a['Predicted Work Rate'] - a['Actual Work Rate'])
a['Signed Delta'] = a['Predicted Work Rate'] - a['Actual Work Rate']
print(a)

# COMMAND ----------

# Merges job data to look at characteristics associated with predictions
eda_df = a.merge(df5, left_on='JOB_ID', right_index = True, how = 'left').sort_values('Delta', ascending = False)
eda_df2 = eda_df[eda_df['Actual Work Rate']==0]
print(list(eda_df2['JOB_ID']))
eda_df

# COMMAND ----------

#Used to see if job size impacted model accuracy.  Used this to determine jobs with 1 or 2 people were throwing off the model.
sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="JOB_NEEDED_ORIGINAL_COUNT",
                    y="Actual Work Rate",
                    hue = 'Delta',
                    data=eda_df)
plt.show()

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Actual Work Rate",
                    y="Predicted Work Rate",
                    hue = 'Delta',
                    data=eda_df)
plt.show()

# COMMAND ----------

sns.histplot(data = eda_df['Signed Delta'], kde = True)
plt.show()

# COMMAND ----------

accuracy_tracker = []
for col in categorical_cols:
  for cat in list(np.unique(eda_df[col])):
    print("Test Performance for: " + col+" "+ cat)
    new_df = eda_df[eda_df[col] == cat]
    size = len(new_df)
    rmse, mae, r2= eval_metrics(new_df['Actual Work Rate'], new_df['Predicted Work Rate'])
    accuracy_tracker.append([col, cat, size, rmse, mae, r2]) 
    # Print out model metrics
tracker_df = pd.DataFrame(accuracy_tracker, columns=['Column', 'Category', 'Size', 'RMSE', 'MAE', 'R2'])
tracker_df.sort_values(by = 'Size', inplace = True)
display(tracker_df)

# COMMAND ----------


