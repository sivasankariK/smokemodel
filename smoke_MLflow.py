# Databricks notebook source
import sys
print(sys.executable)

# COMMAND ----------


!python --version

# COMMAND ----------

# MAGIC %pip install mlflow==1.30.0

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

 


# COMMAND ----------

import pandas as pd
import numpy as np
data = pd.read_csv(r'/dbfs/FileStore/tables/demodatasmoke.csv')

data.head()

# COMMAND ----------

data.describe()

# COMMAND ----------

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

# COMMAND ----------

##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# COMMAND ----------

with mlflow.start_run():
  
  # dtc = LGBMClassifier()
  dtc = XGBClassifier()  
  dtc.fit(X_train, y_train)
  y_pred_class = dtc.predict(X_test)
  accuracy = metrics.accuracy_score(y_test, y_pred_class)
  
  print(accuracy)
  
#   mlflow.log_param('random_state', 10)
#   mlflow.log_param('max_depth', 1)
  mlflow.log_metric('accuracy', accuracy)
  mlflow.sklearn.log_model(dtc, 'model')
  modelpath = "/dbfs/FileStore/tables/demodatasmoke/model-%s-%f" % ("xgb",18)
  # sklearn generates a pickle file and saves it for the model
  mlflow.sklearn.save_model(dtc, modelpath)
  run_id = mlflow.active_run().info.run_id

# COMMAND ----------

   import matplotlib.pyplot as plt
   from sklearn.metrics import plot_confusion_matrix
   plot_confusion_matrix(dtc, X_test, y_test)
   

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn import metrics
metrics.plot_roc_curve(dtc, X_test, y_test) 

# COMMAND ----------

mlflow.search_runs()

# COMMAND ----------

model_uri = 'runs:/'+run_id+'/model'

# COMMAND ----------

model_name = "demomodel"

# COMMAND ----------

artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production',
)
