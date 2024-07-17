# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ast
from collections import Counter

# COMMAND ----------

options = {
  "sfUrl": "vha09841.snowflakecomputing.com",
  "sfUser": "svc_user_prod_app_databricks",
  "SFPassword": "4fzXsWtmhDTF7EpjY5NNffHK",
  "sfDataBase": "BLUECREW",
  "sfSchema": "DM",
  "sfWarehouse": "COMPUTE_WH"
}

start_date = '2023-01-01'
end_date = '2023-11-01'

startdate = pd.to_datetime(start_date).date()
enddate = pd.to_datetime(start_date).date()

# COMMAND ----------

# I can't get this part of the query to work, but I think that's ok.
# query = f"""
# WITH with calendar as (SELECT DATEADD(DAY, SEQ4(), to_date('{start_date}')) AS date
#                   FROM TABLE (GENERATOR(ROWCOUNT => 10000)) -- Number of days after reference date in previous line
#                   WHERE date <= to_date('{end_date}'))
# select * from calendar
# """
# sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# COMMAND ----------

query = f'''
WITH cm_job_shifts as (
    Select distinct
    tsa.START_TIME::date                                                                                                         as date,
    BLUECREW.MYSQL_BLUECREW.REGIONS.LABEL                                                                                as REGION_NAME,
    BLUECREW.DM.DM_COMPANIES.COMPANY_NAME,
    easc.USER_ID,
    tsa.JOB_ID,
    tsa.SEGMENT_INDEX,
    jo.JOB_TYPE,
    jo.JOB_TITLE,
    jo.JOB_CREATED_AT,
    jo.JOB_START_DATE_TIME,
    jo.JOB_STATUS,
    jo.JOB_IS_APPLICATION,
    jo.INVITED_WORKER_COUNT,
    jo.JOB_OVERFILL,
    jo.JOB_NEEDED_LAST_COUNT,
    easc.APPLIED_STATUS_START_DATE                                                                                       as APPLIED_TIME,
    dense_rank() over (partition by easc.USER_ID, tsa.JOB_ID, tsa.SEGMENT_INDEX order by easc.APPLIED_STATUS_START_DATE) as APPLIED_INDEX,---to find first applied time
    tsa.START_TIME                                                                                                       as START_TIME,
    timediff(minute, easc.APPLIED_STATUS_START_DATE, tsa.START_TIME) /60                                                 as time_difference,
    case  when tsa.START_TIME < easc.END_DATE ---were they in status 0 at the start of the shift?
    then 'YES'
    else NULL end                                                                                                   as Assigned_at_Job_Start
from BLUECREW.DM.DM_JOB_TIME_SEGMENTS tsa
  INNER JOIN DM.DM_CM_JOB_APPLIED_HISTORY easc
      on easc.JOB_ID = tsa.JOB_ID
      and tsa.START_TIME >= easc.APPLIED_STATUS_START_DATE ---accepted on the job at/before the start of the shift--todo: caveat
      and dateadd(day, -1, tsa.START_TIME) < easc.END_DATE ---still accepted at the -24 hour mark (policy time) or SNA--todo: caveat
  left join BLUECREW.DM.DM_JOBS jo on jo.JOB_ID = tsa.JOB_ID
  left join BLUECREW.DM.DM_COMPANIES
      on BLUECREW.DM.DM_COMPANIES.COMPANY_ID = jo.COMPANY_ID
                                        left join BLUECREW.MYSQL_BLUECREW.REGIONS
                                                on BLUECREW.MYSQL_BLUECREW.REGIONS.id =
                                                    BLUECREW.DM.DM_COMPANIES.COMPANY_REGION_ID
                                        where 1 = 1
                                          and tsa.ACTIVE = true
                                          and tsa.SHIFT_SEQUENCE = 1        ---first shifts only--todo: caveat
                                          and tsa.CREATED_AT < easc.APPLIED_STATUS_START_DATE---shift created before accepted on the job (shift edits)--todo: caveat
                                          and easc.APPLIED_STATUS_ENUM = 0---ENUM "SUCCESS"
    ),
cm_confirmed as (select distinct
                          ejc.USER_ID,
                          ejc.JOB_ID,
                          to_timestamp_ntz(ejc.UPDATED_AT) as CONFIRM_TIME
                      from BLUECREW.MYSQL_BLUECREW._EVENT_JOB_CONFIRM ejc
                      inner join cm_job_shifts js on js.JOB_ID = ejc.JOB_ID
                          and js.USER_ID = ejc.USER_ID
                          and js.applied_index = 1
                      where 1 = 1
                        and ejc.STATUS = 2--user Affirmed (Confirmed) the job
     ),
/* who came from the waitlist */
     cm_waitlist                      as (select distinct
                                              efw.USER_ID,
                                              efw.JOB_ID,
                                              to_timestamp_ntz(efw.UPDATED_AT) as CONFIRM_TIME
                                          from BLUECREW.MYSQL_BLUECREW._event_add_from_waitlist efw
                                          inner join cm_job_shifts js on js.JOB_ID = efw.JOB_ID
                                              and js.USER_ID = efw.USER_ID
                                              and js.applied_index = 1
                                          where 1 = 1
     ),
/* Who worked the shift (Approved Hours) */
     cm_worked                        as (Select distinct
                                              uh.USER_ID,
                                              uh.JOB_ID,
                                              uh.SHIFT as SEGMENT_INDEX,
                                              to_timestamp_ntz(uh.SHIFT_START) as CLOCK_IN_TIME,---any clock in?---todo: is this useful?
                                              case when uh.STATUS = 0 ---ENUM Approved Hours
                                                   then to_timestamp_ntz(uh.SHIFT_START)---approved hours clock in
                                                   else NULL
                                                   end as WORKED_CLOCK_IN_TIME
                                          from BLUECREW.MYSQL_BLUECREW.USERS_HOURS uh
                                          inner join cm_job_shifts js on js.JOB_ID = uh.JOB_ID
                                              and js.USER_ID = uh.USER_ID
                                              and js.SEGMENT_INDEX = uh.SHIFT
                                              and js.applied_index = 1
                                          where 1 = 1
                                          and uh.ENABLED = 1),
/* SNC */
     snc                              as (Select distinct
                                              snc.USER_ID,
                                              tsa.JOB_ID,
                                              tsa.SEGMENT_INDEX,
                                              snc.CANCEL_TIMESTAMP_UTC as SNC_TIMESTAMP,
                                              dense_rank() over (partition by snc.USER_ID, tsa.JOB_ID, tsa.SEGMENT_INDEX order by snc.CANCEL_TIMESTAMP_UTC) as SNC_INDEX---there can be more than 1 SNC
                                          from BLUECREW.DM.DM_CM_SHORT_NOTICE_CANCELLATIONS snc
                                          INNER JOIN BLUECREW.DM.DM_JOB_TIME_SEGMENTS tsa on tsa.JOB_ID = snc.JOB_ID
                                              and snc.CANCEL_TIMESTAMP_UTC between dateadd(hours, -17, tsa.START_TIME) and tsa.START_TIME ---todo: assumption
                                          inner join cm_job_shifts js on js.JOB_ID = tsa.JOB_ID
                                              and js.USER_ID = snc.USER_ID
                                              and js.SEGMENT_INDEX = tsa.SEGMENT_INDEX
                                              and js.applied_index = 1
                                          where 1 = 1),
/* NCNS */
     ncns                             as (Select distinct
                                              ncns.USER_ID,
                                              ncns.JOB_ID,
                                              ncns.SEGMENT_INDEX,
                                              ncns.NO_SHOW_SHIFT_START_TIME_UTC as NCNS_TIMESTAMP
                                          from BLUECREW.DM.DM_CM_NO_CALL_NO_SHOW ncns
                                          inner join cm_job_shifts js on js.JOB_ID = ncns.JOB_ID
                                              and js.USER_ID = ncns.USER_ID
                                              and js.SEGMENT_INDEX = ncns.SEGMENT_INDEX
                                              and js.applied_index = 1
                                          where 1 = 1),
/* Cancellations: Work Place, Policy, and CM */
     cancel                           as (Select---assignment cancellations
                                              ejc.USER_ID,
                                              js.JOB_ID,
                                              js.SEGMENT_INDEX,
                                              case
                                                  when ejc.REASON = 'other:confirmationPolicy'
                                                      then ejc.CREATED_AT::timestamp_ntz
                                                  else NULL
                                                  end                       as POLICY_CANCEL_TIMESTAMP,
                                              case
                                                  when ejc.REASON = 'workplace-cancellation:null'
                                                      then ejc.CREATED_AT::timestamp_ntz
                                                  else NULL
                                                  end                       as WP_CANCEL_TIMESTAMP,
                                              case
                                                  when ejc.REASON in ('other:confirmationPolicy', 'workplace-cancellation:null')
                                                      then NULL
                                                  else ejc.CREATED_AT::timestamp_ntz
                                                  end                       as CM_CANCEL_TIMESTAMP---ejc.CREATED_AT::timestamp_ntz
                                          from BLUECREW.MYSQL_BLUECREW._EVENT_JOB_CANCEL ejc
                                          inner join cm_job_shifts js on js.JOB_ID = ejc.JOB_ID
                                              and js.USER_ID = ejc.USER_ID
                                              and js.applied_index = 1
                                          where 1 = 1
                                            and ejc.CREATED_AT::timestamp_ntz < js.START_TIME---cancel before start of shift
                                          union
                                          Select---single shift cancellations
                                              uh.USER_ID,
                                              uh.JOB_ID,
                                              uh.SHIFT                     as SEGMENT_INDEX,
                                              NULL as POLICY_CANCEL_TIMESTAMP,
                                              case
                                                  when uh.note = 'Workplace cancellation' then uh.UPDATED_AT::timestamp_ntz
                                                  else NULL
                                                  end                      as WP_CANCEL_TIMESTAMP,
                                              case
                                                  when uh.note = 'Workplace cancellation' then NULL
                                                  else uh.UPDATED_AT::timestamp_ntz
                                                  end                      as CM_CANCEL_TIMESTAMP
                                          from BLUECREW.MYSQL_BLUECREW.USERS_HOURS uh
                                          inner join cm_job_shifts js on js.JOB_ID = uh.JOB_ID
                                              and js.USER_ID = uh.USER_ID
                                              and js.SEGMENT_INDEX = uh.SHIFT
                                              and js.applied_index = 1
                                          where 1 = 1
                                            and uh.ENABLED = 1
                                            and uh.status = 3),
/* accepted at job start: look at any applied_index */
    accepted_at_job_start as (
        Select distinct
               j.USER_ID,
               j.JOB_ID,
               j.SEGMENT_INDEX,
               j.Assigned_at_Job_Start
        from cm_job_shifts j
        where 1=1
        and j.Assigned_at_Job_Start = 'YES'
    ),
/* Detail Data */
raw_data as (select
                 j.date,
--                 weekofyear('2023-04-22') as week,
                 j.REGION_NAME,
                 j.COMPANY_NAME,
                 j.USER_ID,
--                 u.USER_STATUS,
                 u.USER_FULL_NAME,
                 case
                     when j.time_difference < 24 then 'Accept < 24 hours'
                     when j.time_difference between 24 and 48 then 'Accept 24 to 48 hours'
                     when j.time_difference > 48 then 'Accept > 48 hours'
                     end                                                                                                                                  as Accept_Confirm_First_Shift_Group,
                 j.JOB_ID,
                 j.JOB_TYPE,
                 j.JOB_TITLE,
                 j.JOB_CREATED_AT,
                 j.JOB_START_DATE_TIME,
                 timediff(minute, j.JOB_CREATED_AT, j.JOB_START_DATE_TIME) /(60*24)                                                                       as sourcing_time_diff_days,
--                 j.JOB_STATUS,
                 j.SEGMENT_INDEX,
                 j.START_TIME                                                                  as SHIFT_START_TIME,
                 j.APPLIED_TIME,
                 j.JOB_IS_APPLICATION,
                 j.INVITED_WORKER_COUNT,
                 j.JOB_NEEDED_LAST_COUNT,
                 j.JOB_OVERFILL,
                 j.time_difference                                                                                                                        as apply_time_diff,
                 c.CONFIRM_TIME,
                 w.CONFIRM_TIME                                                                                                                           as FROM_WAITLIST_TIME,
                 timediff(minutes, c.CONFIRM_TIME, j.START_TIME) / 60                                                                                     as confirm_time_diff,
                 dense_rank() over (Partition by j.USER_ID, j.JOB_ID, j.SEGMENT_INDEX order by timediff(minutes, c.CONFIRM_TIME, j.START_TIME) / 60 desc) as confirm_index,---can be more than 1
                 acc.Assigned_at_Job_Start,
--                  s.CLOCK_IN_TIME,
                 s.WORKED_CLOCK_IN_TIME,
                 snc.SNC_TIMESTAMP,
                 ncns.NCNS_TIMESTAMP,
                 cancel.POLICY_CANCEL_TIMESTAMP,
                 cancel.WP_CANCEL_TIMESTAMP,
                 case when snc.SNC_TIMESTAMP is null
                          then cancel.CM_CANCEL_TIMESTAMP
                     end as CM_CANCEL_TIMESTAMP,
                 u.USER_PROFILE_LINK
             from cm_job_shifts j
             INNER JOIN BLUECREW.DM.DM_USERS_DATA u on u.USER_ID = j.USER_ID
             left join cm_confirmed c on c.USER_ID = j.USER_ID
                 and c.JOB_ID = j.JOB_ID
                 and c.CONFIRM_TIME <= j.START_TIME
             left join cm_waitlist w on w.USER_ID = j.USER_ID
                 and w.JOB_ID = j.JOB_ID
                 and w.CONFIRM_TIME <= j.START_TIME
                 and c.USER_ID is null---not confirmed---todo: assumption
             left join ncns ncns on ncns.JOB_ID = j.JOB_ID
                 and ncns.USER_ID = j.USER_ID
                 and ncns.SEGMENT_INDEX = j.SEGMENT_INDEX
             left join cm_worked s on s.USER_ID = j.USER_ID
                 and s.JOB_ID = j.JOB_ID
                 and s.SEGMENT_INDEX = j.SEGMENT_INDEX
                 and ncns.USER_ID is null---not a NCNS---todo: assumption
             left join cancel on cancel.JOB_ID = j.JOB_ID
                 and cancel.USER_ID = j.USER_ID
                 and cancel.SEGMENT_INDEX = j.SEGMENT_INDEX
                --  and s.WORKED_CLOCK_IN_TIME is null---todo: assumption - wrong can work after policy removal
             left join snc snc on snc.JOB_ID = j.JOB_ID
                 and snc.USER_ID = j.USER_ID
                 and snc.SEGMENT_INDEX = j.SEGMENT_INDEX
                 and snc.SNC_INDEX = 1---can have multiple SNC Entries
                 and ncns.USER_ID is null---not a NCNS---todo: assumption
                 and cancel.CM_CANCEL_TIMESTAMP is not null--not a WP or Policy Cancel ---todo: assumption
                 and s.WORKED_CLOCK_IN_TIME is null---todo: assumption
             left join accepted_at_job_start acc on acc.JOB_ID = j.JOB_ID
                 and acc.USER_ID = j.USER_ID
                 and acc.SEGMENT_INDEX = j.SEGMENT_INDEX
             where 1 = 1
               and j.APPLIED_INDEX = 1
                              ),
    total_cm                         as (select
                                              weekofyear(j.date)                                                    as week,
                                              count(distinct j.USER_ID || '-' || j.JOB_ID || '-' ||j.SEGMENT_INDEX) as total_applied_users
                                          from cm_job_shifts j
                                          where 1 = 1
                                          group by 1),
    /* list all Accept_Confirm_First_Shift_Groups to display zeros */
     Accept_Confirm_First_Shift_Group as (select 'Accept < 24 hours' as Accept_Confirm_First_Shift_Group, 1 as index
                                          union
                                          select 'Accept 24 to 48 hours' as Accept_Confirm_First_Shift_Group, 2 as index
                                          union
                                          select 'Accept > 48 hours' as Accept_Confirm_First_Shift_Group, 3 as index),
    /* get the job needed on a date*/
    needed_cm as (
        Select distinct j.date,
    j.JOB_ID,
    j.SEGMENT_INDEX,
    needed.NEEDED as needed
from cm_job_shifts j
    left join dm.DM_JOB_NEEDED_HISTORY needed on needed.JOB_ID = j.JOB_ID
    and j.date >= needed.START_DATE
    and j.date < needed.END_DATE
where 1=1
    ),
    /* get the weekly needed totals */
    needed_cm_agg as (
select weekofyear(n.date) as week,
    sum(n.needed) as total_needed
from needed_cm n
where 1=1
group by 1
    )
Select * from raw_data
'''
sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# COMMAND ----------

raw_data = sdf.toPandas()
raw_data

# COMMAND ----------

#Some cleaning to determine if people worked and change apply time into days and truncated days for visualization below

raw_data['Worked?']=raw_data['WORKED_CLOCK_IN_TIME'].apply(lambda x: "N" if x!=x else "Y")
raw_data["APPLY_TIME_DIFF_DAYS"]=raw_data["APPLY_TIME_DIFF"].apply(lambda x: x/24 if x/24 <30 else 30)
raw_data["APPLY_TIME_DIFF_DAYS_TRUNC"] = raw_data["APPLY_TIME_DIFF_DAYS"].astype(int)

raw_data["SOURCING_TIME_DIFF_DAYS"]=raw_data["SOURCING_TIME_DIFF_DAYS"].apply(lambda x: x if x <30 else 30)
raw_data["SOURCING_TIME_DIFF_DAYS_TRUNC"] = raw_data["SOURCING_TIME_DIFF_DAYS"].astype(int)

#It seems like there are a ton of people who have an applied_time = shift_start_time.  This seems like bad data because some of them have a confirmation time that is before the applied_time
raw_data_0_dropped = raw_data[raw_data["APPLY_TIME_DIFF_DAYS"]!=0]


#limiting the dataset to FN_Logistics. It looks like if invite_worker_count>0 then apply_time_diff = 0?  Should investigate
fn_logistics = raw_data[(raw_data["COMPANY_NAME"]=="FN Logistics Inc.")&(raw_data["APPLY_TIME_DIFF"]!=0)]
fn_logistics["APPLY_TIME_DIFF_DAYS"]=fn_logistics["APPLY_TIME_DIFF"]/24
#almost half of the entries have an apply_time = to the shift_start_time
fn_logistics_apply0 = raw_data[(raw_data["COMPANY_NAME"]=="FN Logistics Inc.")&(raw_data["APPLY_TIME_DIFF"]==0)]
# fn_logistics
fn_logistics.sort_values(by="JOB_ID", inplace = True)
fn_logistics

# COMMAND ----------

# MAGIC %md
# MAGIC There are 14 people in the user_data that are duplicates because they have different marketing campaign IDs.  
# MAGIC Commented out since it doesn't impact the code below.

# COMMAND ----------

# query = f'''
# /* Detail Data */
# with a as(
# select
#   u.USER_ID,
#   count(*) as user_count
#   from BLUECREW.DM.DM_USERS_DATA u
#   group by 1
#   order by user_count desc)
#    select distinct USER_ID, USER_MARKETING_CAMPAIGN_ID
#     from BLUECREW.DM.DM_USERS_DATA u
#    where u.USER_ID in (select USER_ID from a where user_count = 2)
#    order by 1
# '''
# sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# COMMAND ----------

# user_data = sdf.toPandas()
# user_data

# COMMAND ----------

sns.histplot(data = raw_data_0_dropped, x = "SOURCING_TIME_DIFF_DAYS", hue="Worked?", bins = 30)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Cancellation Rates
# MAGIC It looks like there may be different cancellation rates based on time horizon when looking at FN logistics, but not for the raw data

# COMMAND ----------

# All data with 0 dropped
sns.histplot(data = raw_data_0_dropped, x = "APPLY_TIME_DIFF_DAYS", hue="Worked?", bins = 30)
plt.show()

# COMMAND ----------

# Calculating the percent of people that worked based on when they applied
result = raw_data.groupby(['APPLY_TIME_DIFF_DAYS_TRUNC', 'Worked?']).size().reset_index(name='Count')

# Pivot the data for stacked bar plot
pivot_result = result.pivot(index='APPLY_TIME_DIFF_DAYS_TRUNC', columns='Worked?', values='Count').fillna(0)

# Calculate percentages
pivot_result_percentage = pivot_result.div(pivot_result.sum(axis=1), axis=0) * 100
pivot_result_percentage

# COMMAND ----------

# FN_Logistics
sns.histplot(data = fn_logistics, x = "APPLY_TIME_DIFF_DAYS", hue="Worked?", bins = 30)
plt.show()

# COMMAND ----------

# Calculating the percent of people that worked based on when they applied
result = fn_logistics.groupby(['APPLY_TIME_DIFF_DAYS_TRUNC', 'Worked?']).size().reset_index(name='Count')

# Pivot the data for stacked bar plot
pivot_result = result.pivot(index='APPLY_TIME_DIFF_DAYS_TRUNC', columns='Worked?', values='Count').fillna(0)

# Calculate percentages
pivot_result_percentage = pivot_result.div(pivot_result.sum(axis=1), axis=0) * 100
pivot_result_percentage

# COMMAND ----------

sns.histplot(data = fn_logistics, x = "SOURCING_TIME_DIFF_DAYS", bins = 30)
plt.show()

# COMMAND ----------

sns.histplot(data = fn_logistics, x = "SOURCING_TIME_DIFF_DAYS", hue="Worked?", bins = 30)
plt.show()

# COMMAND ----------

# Calculating the percent of people that worked based on when they applied
result = fn_logistics.groupby(['SOURCING_TIME_DIFF_DAYS_TRUNC', 'Worked?']).size().reset_index(name='Count')

# Pivot the data for stacked bar plot
pivot_result = result.pivot(index='SOURCING_TIME_DIFF_DAYS_TRUNC', columns='Worked?', values='Count').fillna(0)

# Calculate percentages
pivot_result_percentage = pivot_result.div(pivot_result.sum(axis=1), axis=0) * 100
pivot_result_percentage

# COMMAND ----------

job_columns = ["REGION_NAME",	"COMPANY_NAME","JOB_ID",	"JOB_TYPE",	"JOB_TITLE",	"JOB_CREATED_AT",	"JOB_START_DATE_TIME", "SOURCING_TIME_DIFF_DAYS", "SOURCING_TIME_DIFF_DAYS_TRUNC", "JOB_NEEDED_LAST_COUNT"]
job_duplicates_dropped = raw_data[job_columns].drop_duplicates()
job_duplicates_dropped

# COMMAND ----------

sns.histplot(data = job_duplicates_dropped, x = "SOURCING_TIME_DIFF_DAYS", bins = 30)
plt.show()

# COMMAND ----------

fn_logistics_jobs = job_duplicates_dropped[job_duplicates_dropped["COMPANY_NAME"]=="FN Logistics Inc."]
fn_logistics_jobs

# COMMAND ----------

sns.histplot(data = fn_logistics_jobs, x = "SOURCING_TIME_DIFF_DAYS", bins = 30)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Interesting Job Example
# MAGIC It appears this job was created because the person may have been there already.  I suspect that they showed up and the FN_Logistics created a shift rather than send them home.  How often does this occur?

# COMMAND ----------

# job_242590 = raw_data[(raw_data["JOB_ID"]==242590) &(raw_data["APPLY_TIME_DIFF"]==0)]
job_276274 = raw_data[(raw_data["JOB_ID"]==276274)]
job_276274

# COMMAND ----------

short_notice_jobs = raw_data[(raw_data["SOURCING_TIME_DIFF_DAYS"]<=0.05)&(raw_data["DATE"]>=startdate)&(raw_data["COMPANY_NAME"]=="FN Logistics Inc.")]
short_notice_jobs

# COMMAND ----------

fn_logistics_apply0


# COMMAND ----------

# MAGIC %md
# MAGIC # FN_Logistics_Inc fill rates and quantity

# COMMAND ----------

query = f'''
SELECT
    dm_companies."COMPANY_NAME"  AS "dm_companies.company_name",
        CASE WHEN COALESCE(SUM(fact_time_to_fill."JOB_NEEDED_LAST_COUNT" ), 0) = 0 THEN NULL ELSE COALESCE(SUM(fact_time_to_fill."SIGN_UP_JOIN_COUNT" ), 0) / COALESCE(SUM(fact_time_to_fill."JOB_NEEDED_LAST_COUNT" ), 0) END AS fill_percent,
        COALESCE(SUM(fact_time_to_fill."SIGN_UP_JOIN_COUNT" ),0) as total_fills,
        COALESCE(SUM(fact_time_to_fill."JOB_NEEDED_LAST_COUNT" ),0) as total_jobs_to_fill
FROM "DM"."DM_JOBS"
     AS jobs
INNER JOIN "DM"."DM_COMPANIES"
     AS dm_companies ON (jobs."COMPANY_ID") = (dm_companies."COMPANY_ID")
LEFT JOIN "DM"."FACT_TIME_TO_FILL"
     AS fact_time_to_fill ON (jobs."JOB_ID") = (fact_time_to_fill."JOB_ID")
WHERE (((((( jobs."JOB_CREATED_AT"  ))) >= DATEADD('month', -10, DATE_TRUNC('month', CURRENT_DATE())) AND ((( jobs."JOB_CREATED_AT"  ))) < ((DATEADD('month', 11, DATEADD('month', -10, DATE_TRUNC('month', CURRENT_DATE())))))))) AND (dm_companies."COMPANY_NAME" ) = 'FN Logistics Inc.' AND (dm_companies."COMPANY_REGION" ) IS NOT NULL
GROUP BY
    1
HAVING CASE WHEN COALESCE(SUM(fact_time_to_fill."JOB_NEEDED_LAST_COUNT" ), 0) = 0 THEN NULL ELSE COALESCE(SUM(fact_time_to_fill."SIGN_UP_JOIN_COUNT" ), 0) / COALESCE(SUM(fact_time_to_fill."JOB_NEEDED_LAST_COUNT" ), 0) END IS NOT NULL
ORDER BY
    2
'''
sdf = spark.read.format("snowflake").options(**options).option("query", query).load()

# COMMAND ----------

data = sdf.toPandas()
data

# COMMAND ----------


