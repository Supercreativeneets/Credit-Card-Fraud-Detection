# Credit Card Fraud Detection - About the Project
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## Dataset
The dataset has been collected and analyzed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. The dataset could be found in https://www.kaggle.com/mlg-ulb/creditcardfraud

This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
        
## **Workflow**
    
    
### **Exploring the dataset**
Understanding Class-Imbalance
    
From our analysis, we observe there is a lot of imbalance in the classes, with most of the transactions were Non-Fraud (99.83%) of the time, while Fraud transactions occur (0.17%) of the time in the dataframe.

Using this imbalanced data as such is not a good idea for training a model to classify if a transaction is fraudulent or not. This is because, if this imbalanced data is used to train a model, the algorithm does not have a decent amount of fraudulent-data to learn the patterns of fraudulent transactions. Thus, it most probably assumes that every transaction is non-fraudulent(the dominant class of the data).

This would be a pity because the model naively assumes but doesn't learn/detect the patterns in order to classify.

The solution : To make the dataset balanced, we could either undersample or oversample it.

* Under-sampling: In undersampling, we reduce the dataset such that the number of samples of one class is to that of the other class. But this method has a trade-off with the amount of information lost in the form of the samples removed.
* Over-sampling: Next is the oversampling technique. We increase the number of total samples in the dataset by generating the synthetic samples for the minority class in order to achieve the balance between both the classes. The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. Instead, new examples can be synthesized from the existing examples. This is a type of data augmentation for the minority class and is referred to as the Synthetic Minority Oversampling Technique, or SMOTE for short.

### **Train-test split
The data is split into train and test parts, in order to ***prevent any data leakage*** and to keep the test data untouched, before oversampling.

### Feature Scaling
We have scaled the Amount and Time features using ***StandardScaler***.

### Apply SMOTE
We then ***applied the SMOTE technique to oversample the train data*** and formed a new dataset with the thus obtained over-sampled datapoints.

### Cross Validation (Grid Search CV)
GridSearchCV (Cross-Validation) is a technique used to tune hyperparameters for machine learning models. It systematically searches through a specified set of hyperparameter combinations and evaluates the model's performance using cross-validation to find the best combination of hyperparameters that produces the optimal model.

### Build and train model (Logistic Regression Classifier)
GridSearch CV method is used to train logistic regression classifiers with the different combinations of these parameters, and got the best logistic regression classifier which yields the least loss on the over-sampled data-set.

### Model Evaluation
Best estimator thus obtained is used to evaluate its performance on the unseen test data. We calculated the recall, confusion-matrix and roc-auc scores.

