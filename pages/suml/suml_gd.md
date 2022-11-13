---
title: Gradient Descent
tags: [formatting]
keywords: notes, gd
last_updated: 13.Nov.2022
summary: Learning note for gradient descent
sidebar: mydoc_sidebar
permalink: suml_gd.html
folder: pages/suml
---

## Normal Equation for min Cost function
OLS:
#normalequation 
	minimize the sum of squared residuals.
	$e^{2}_1+e^{2}_2+...+e^{n}_1$

	$b=(X^TX)^{-1}X^{T}y$

	㊟inverting a metric is heavy computing $O(n^{3})$

## **Gradient Descent** (GD)
#gradient_descent 

	- iterative optimization algorithm. 
	- characteristic: Endpoint id dependent on the initial starting point

## Gradient Descent in a nutshell: 
1. Scale data
2. Start at random point $J(b)$
3. Search for min $J(b)$, untill you are somewhere flat
	1) Start with some random parameter $b$
	2) Start descent:
		- Set learning rate (step-size)
		- Adjust your parameters (step)
	3) Repeat **2** till there is no further improvement 
4. Keep changing $b$ to reduce $J(b)$ till we hopefully get to a minimum

### WHEN to use:
	- Simple optimization procedure that can be used for many machine learning
	- used for "Backpropagation" in Neural Nets
	- used for online learning
	- gives faster results for problems with many features

## Cost functions
#cost_functions

$J(b0,b1)=\frac{1}{2n}\sum^{n}_{i=1}(\hat{y_i}-y_i)$

## Learning Rate
#step #learning_rate
Choose a proper learning rate

## GD main challenges
#convex #plateaus #scale

### Problems
- **Local minima** for non-convex function: 
	-usually good enough. and we end up at local minima.
-  **plateaus**: 
	happens more often than local minima due to many subtle variables. Methods to get over Momentum:
	* momentum(2nd derivatives): how are the last n steps descent?
- **Data scaling**: single learning rate varies for different variables.

## Three variants of GD
#normalequation #batch_gd #sgd #mini_batch
### Batch GD
- All instances are used
- Faster than the Normal Equations

### Stochastic GD (drunk but fast)
- Only ONE random instances
- Much faster, suitable for large data sets
- Online training possible

### Mini-Batch GD
- Small random subset
- Performs better than SGD
- Drawback: harder not to end up in local minima

