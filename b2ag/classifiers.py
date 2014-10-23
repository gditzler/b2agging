#!/usr/bin/env python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

__author__ = "Gregory Ditzler"
__copyright__ = "Copyright 2014, Gregory Ditzler"
__credits__ = "Gregory Ditzler"
__license__ = "GPL V3.0"
__version__ = "0.1.0"
__status__ = "development"
__email__ = "gregory.ditzler@gmail.com"


class base_model:
  """
  model()

  `model` is a generic class for constructing a base prediction model that 
  is derived from the sklearn library. It simply acts as a wrapper for the 
  ease of experimentation. 

  Methods
    __init__(self, params)
    fit(self, data, labels)
    predict(self, data, probs=False)
    predict_error(self, data, labels)
  """
  def __init__(self, params):
    """
    __init__(self, params)
      @params: dictionary containing the parameters for a classification model.
        the keys in the params dictionary are identical to the parameters that
        is used in the sklearn model with the exception of an added type field
        which indicates the classifier. currently, the following classifiers 
        are implemented.
        1) `lr`: logistic regression
        2) `cart`: decision tree
    """
    self.params = params
    self.mdl = None

  def fit(self, data, labels):
    """
    fit(self, data, labels)
    @self.params: parameters specific for a classifier
    @data: feature variables (len(data) = n_observations)
    @labels: class labels (len(labels) = n_observations)
    """

    # perform some high level error checking here
    if not self.params.has_key("type"):
      raise ValueError("Classifier type must be set.")
    elif len(data) != len(labels):
      raise ValueError("Data & labels must be the same length")

    if self.params["type"] == "lr":
      # set missing parameters for logisic regression
      if not self.params.has_key("penalty"):
        self.params["penalty"] = "l2"
      if not self.params.has_key("dual"):
        self.params["dual"] = False
      if not self.params.has_key("C"):
        self.params["C"] = 1.
      if not self.params.has_key("fit_intercept"):
        self.params["fit_intercept"] = True
      if not self.params.has_key("intercept_scaling"):
        self.params["intercept_scaling"] = 1
      if not self.params.has_key("tol"):
        self.params["tol"] = 0.0001

      clfr = LogisticRegression(penalty=self.params["penalty"], dual=self.params["dual"],
                               tol=self.params["tol"], C=self.params["C"],
                               fit_intercept=self.params["fit_intercept"],
                               intercept_scaling=self.params["intercept_scaling"],
                               class_weight=None, random_state=None)
      self.mdl = clfr.fit(data, labels)
    elif self.params["type"] == "cart":
      #
      if not self.params.has_key("criterion"):
        self.params["criterion"] = "gini"
      if not self.params.has_key("splitter"):
        self.params["splitter"] = "best"
      if not self.params.has_key("max_depth"):
        self.params["max_depth"] = None
      if not self.params.has_key("min_samples_split"):
        self.params["min_samples_split"] = 2
      if not self.params.has_key("min_samples_leaf"):
        self.params["min_samples_leaf"] = 1
      if not self.params.has_key("max_features"):
        self.params["max_features"] = None
      if not self.params.has_key("random_state"):
        self.params["random_state"] = None
      if not self.params.has_key("min_density"):
        self.params["min_density"] = None
      if not self.params.has_key("compute_importances"):
        self.params["compute_importances"] = None
      if not self.params.has_key("max_leaf_nodes"):
        self.params["max_leaf_nodes"] = None

      clfr = DecisionTreeClassifier(criterion=self.params["criterion"],
                                   splitter=self.params["splitter"],
                                   max_depth=self.params["max_depth"],
                                   min_samples_split=self.params["min_samples_split"],
                                   min_samples_leaf=self.params["min_samples_leaf"],
                                   max_features=self.params["max_features"],
                                   random_state=self.params["random_state"],
                                   min_density=self.params["min_density"],
                                   compute_importances=self.params["compute_importances"],
                                   max_leaf_nodes=self.params["max_leaf_nodes"])
      self.mdl = clfr.fit(data, labels)

    elif self.params["type"] == "knn":
      if not self.params.has_key("n_neighbors"):
        self.params["n_neighbors"] = 5
      if not self.params.has_key("weights"):
        self.params["weights"] = "uniform"
      if not self.params.has_key("algorithm"):
        self.params["algorithm"] = "auto"
      if not self.params.has_key("leaf_size"):
        self.params["leaf_size"] = 30
      if not self.params.has_key("p"):
        self.params["p"] = 2
      if not self.params.has_key("metric"):
        self.params["metric"] = "euclidean"
      
      clfr = KNeighborsClassifier(n_neighbors=self.params["n_neighbors"], 
                                  weights=self.params["weights"], 
                                  algorithm=self.params["algorithm"], 
                                  leaf_size=self.params["leaf_size"], 
                                  p=self.params["p"], 
                                  metric=self.params["metric"])
      self.mdl = clfr.fit(data, labels)
    else:
      raise ValueError("Unknown key "+ self.params["type"]+"for classifier type.")
    
    return self

  def predict(self, data, probs=False):
    """
    predict(self, data, probs=False)
      @data
      @probs: probabilities

    Make predictions on data.
    """
    if probs:
      p = self.mdl.predict_proba(data)
    else:
      p = self.mdl.predict(data)
    return p

  def predict_error(self, data, labels):
    """
    predict_error(self, data, labels)
      @data
      @labels

    Compute the error of a prediction. 
    """
    return 1.-accuracy_score(labels, self.predict(data))


class b2ag:
  """
  """
  def __init__(self, n_models=50, params={"type":"lr"}, percent_train=1., 
               percent_eval=1., eval_samples=25):
    """
    """
    self.n_models = 50
    self.params = params
    self.percent_train = percent_train
    self.percent_eval = percent_eval
    self.models = []
    self.eval_samples = eval_samples

  def fit(self, data, labels):
    """
    fit(self, data, labels)

    Fit a b2ag model
    """
    for n in range(self.n_models):
      idx = np.random.randint(low=0, high=len(data), 
                              size=np.floor(self.percent_train*len(data)))
      data_n = data[idx, :]
      labels_n = labels[idx]
      base_n = base_model(self.params)
      self.models.append(base_n.fit(data, labels))

    return self

  def predict(self, data):
    """
    predict(self, data)
      @data

    Make predictions on data using b2ag. Return the confidence level and the 
    standard deviation of the confidence level for the ensemble model. 
    """
    for i in range(self.eval_samples):
      idx = np.random.randint(low=0, high=len(self.models), size=np.floor(self.percent_eval*self.n_models))
      eval_models = [self.models[j] for j in idx]
      for n, model in enumerate(eval_models):
        if n == 0:
          yhat = model.predict(data, probs=True)
        else:
          yhat += model.predict(data, probs=True)
      
      yhat = yhat
      normalizer = np.tile(np.sum(yhat, axis=1), (len(yhat[0]),1)).transpose()
      yhat = yhat/normalizer
      
      if i == 0:
        phat = yhat 
        phat2 = yhat**2
      else:
        phat += yhat
        phat2 += yhat**2
    return phat/self.eval_samples, np.sqrt(np.abs(phat2/self.eval_samples - (phat/self.eval_samples)**2))




