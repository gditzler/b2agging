#!/usr/bin/env python
import numpy as np
import pandas as pd
import sklearn.preprocessing
import pickle
import scipy
from sklearn_pandas import DataFrameMapper
from scipy.io import loadmat

__author__ = "Gregory Ditzler"
__copyright__ = "Copyright 2014, Gregory Ditzler"
__credits__ = "Gregory Ditzler"
__license__ = "GPL V3.0"
__version__ = "0.1.0"
__status__ = "development"
__email__ = "gregory.ditzler@gmail.com"


def load_table(data_fp):
  """
  load_table(data_fp)
    @data_fp

  The assumption is that the last column in the file is 'class'. For example, 
  a basic file might look like. 
  ~~~~~~~~~~~~~~~~~
  feature_1,feature_2,feature_3,class
  1,1,3,1
  12,3,1,2
  ...

  >>> data, labels = load_table(data_fp)
  """
  try: 
    df = pd.read_csv(data_fp)
  except:
    raise IOError("The file:"+str(data_fp)+" does not exist.")

  try:
    dead_var = df["class"] # unused until i find a has_key() function for the DF
  except:
    raise KeyError("Pandas DataFrame must have the `class` key.")

  maps = []
  for key in df.keys():
    if key == "class":
      maps.append( (key, sklearn.preprocessing.LabelBinarizer()) )
    else:
      maps.append( (key, sklearn.preprocessing.StandardScaler()) )
  mapper = DataFrameMapper(maps)
  data_mapped = mapper.fit_transform(df)
  
  n_variables = len(data_mapped[0])
  n_classes = len(np.unique(df["class"]))

  data = data_mapped[:,:n_variables-n_classes]
  labels = np.argmax(data_mapped[:,n_variables-n_classes:], axis=1)
  return data, labels

def load_mat(data_fp):
  """
  load_mat(data_fp)
    @data_fp

  load data from a matlab file. 
  """
  obj = scipy.io.loadmat(data_fp)
  try:
    data = obj["data"]
  except:
    raise KeyError("The field `data` must be in the pickle file.")
  try:
    labels = obj["labels"]
  except:
    raise KeyError("The field `labels` must be in the pickle file.")
  return data, labels

def load_pickle(data_fp):
  """
  load_pickle(data_fp)
    @data_fp

  Load data from a pickle file. The pickled object should have the fields:
  `data` and `labels`. Refer to data/digit-raw.pkl
  """
  obj = pickle.load(open(data_fp, "rb"))
  try:
    data = obj["data"]
  except:
    raise KeyError("The field `data` must be in the pickle file.")
  try:
    labels = obj["labels"]
  except:
    raise KeyError("The field `labels` must be in the pickle file.")
  return data, labels


def load_parameter_file(params_fp):
  """
  load_parameter_file(params_fp)
  @params_fp

  Load a parameters file describing a base classifier. The file should look 
  something like:
  ~~ paramters.txt ~~
  type:lr:string
  tol:0.001:float
  penalty:l2:string
  dual:False:bool

  In general you have: 
  <parameter name>:<value>:<type>

  Available types are:
   - None
   - float
   - int
   - bool
   - string 
  """
  handl = open(params_fp, "U")
  params = {}
  for line in handl:
    
    try:
      line_split = line[:-1].split(":")
    except:
      continue
    if len(line_split) != 3: 
      continue

    if line_split[2] == "None":
      params[line_split[0]] = None
    elif line_split[2] == "float":
      params[line_split[0]] = float(line_split[1])
    elif line_split[2] == "int":
      params[line_split[0]] = int(line_split[1])
    elif line_split[2] == "bool":
      params[line_split[0]] = bool(line_split[1])
    elif line_split[2] == "string":
      params[line_split[0]] = line_split[1]
  return params


