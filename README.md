A RESTFUL API that performs classification algorithm

Project specs:
The goal of this project is to build a Python-based API to perform a classification algorithm and redirect traffic to different webpages based on different class predicted. This API could be accessed through simple GET method. 

About the model:
A consumer clustering was done in R using hierarchical clustering and cluster predicted was stored in the training dataset as "cluster". This project will not focus on clustering part, but rather the typing tool/segment prediction part. Gradient boosted classification algorithm was used here to predict the consumer's cluster. "cluster" variable obtained in clustering stage was treated as dependent variable here. And a group of variable were feeded into model as independent features. Given that this project's goal is to illustrate how to wrap ML model in FLASK API, we will not focus on the feature selection which typically is done before tuning the final model. We used GridsearchCV to tune hyper-parameters.

About the API:
Language:
Written in Python and built using "flask" packages.
Environment:
The API will need to call the following packages: flask, numpy, pandas, json, sklearn.
This API will need to call the boosted tree model saved in ".pkl" format.
Routes:
There is only one route called "predict_segment". If there's need to expand API capability, a separate route could be added.

The correct work flow:
1) Train the cluster prediction model
2) Write API locally (including build local venv and install all dependent packages. For detail please check: http://flask.pocoo.org/docs/0.12/installation/)
3) Test API locally (Deploy API locally and test if the API performs prediction properly. In this stage, you could use a few cases in training dataset as test cases. For details on how to deploy API please check: http://flask.pocoo.org/docs/1.0/quickstart/#debug-mode)
4) Deploy API 



