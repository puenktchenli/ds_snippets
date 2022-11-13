---
title: K-nearest Neighbors Algorithm
tags: [formatting]
keywords: notes, knn
last_updated: 13.Nov.2022
summary: Learning note for knn
sidebar: mydoc_sidebar
permalink: suml_knn.html
folder: pages/suml
---


# *K*-nearest neighbors algorithm (KNN)
---

## Distance Metrics
### Minkowski Distance (most used)
$Minkowski\: Distance = \sqrt[n]{\sum_{i=1}^k |x_i - y_i|^n}$  
$Minkowski\: Distance = \sqrt[n]{\sum_{i=1}^k side\:length^n}$ 

### Euclidean Distance

$\ Euclidean\: Distance = \sqrt{\sum_{i=1}^k |x_i - y_i|^2}$ 
$\ Euclidean\: Distance = \sqrt{\sum_{i=1}^k side\:length^2}$

### Manhattan Distance
$\ Manhattan\: Distance = \sum_{i=1}^k |x_i - y_i|$
$\ Manhattan\: Distance = \sum_{i=1}^k side\: length$

## Pros and Cons

### Pros
* No assumptions about data
* Simple algorithm — easy to understand
* Can be used for classification and regression
### Cons
* High memory requirement — All of the training data must be present in memory in order to calculate the closest K neighbors
* Sensitive to irrelevant features
* Sensitive to the scale of the data since we’re computing the distance to the closest K points

## Usecase
music prediction