# Bank's Customers Churn Prediction

![Reduce-churn-Rate-02](https://github.com/hayasalman/Artificial-Neural-Network-ANN/assets/71796909/488d35c7-d7bd-4900-9fa8-70ec71932126)

 ## Overview 

   A dataset contains some customers who are withdrawing their account from the bank due to some loss and other issues. 
   With this data we try to analyze those customers' behavior and predict who is more likely to churn. It's a binary classification problem where 
   customers either stay with the bank or leave, and for this particular case we will be using the **Artificial Neural Network (ANN)**.

  ### Dataset Features description
  
- **CustomerId**: the customer identifier.
- **Surname**: the surname of a customer.
- **CreditScore**: a score with a range between 300 and 850, a credit score of 700 or above is generally considered good. A score of 800 or above on the same range is considered to be excellent.
- **Geography**: customer location.
- **Gender**: customer gender.
- **Age**: customer age.
- **Tenure**: refers to the number of years that the customer has been a client of the bank.
- **Balance**: the amount of balance in the credit card.
- **NumOfProducts**: refers to the number of products that a customer has purchased through the bank.
- **HasCrCard**: denotes whether or not a customer has a credit card.
- **IsActiveMember**: denotes whether or not a customer is active or not.
- **EstimatedSalary**: the estimated amount of the customer salary.
- **Complain**: customer has complaint or not.
- **Satisfaction Score**: Score provided by the customer for their complaint resolution (1 -5).
- **Card Type**: type of card hold by the customer.
- **Points Earned**: the points earned by the customer for using credit card.
- **Exited**: whether or not the customer left the bank.

## About The Dataset

- In total, this dataset has 10000 observations and 17 features, except it has some missing/nulls that need to be handled before doing any kind of analysis/modeling.
-  Data source stored as CSV file and can be accessed through this link : [Dataset](https://github.com/hayasalman/Artificial-Neural-Network-ANN/blob/main/Customer_Churn_Records.csv)

## Coding
-  Python Integrated Development Environment (IDE) : Google Colab Notebooks

   **Packeges used**
   
  * **pandas - numby** : used for data manipulation.
  * **matplotlib - seaborn** : used for data visualtion.
  * **sklearn** : used to any preprocessing steps required before feeding the dataset into the machine learning algorithm,
   and to evaluate the performance of the models.
  * **tensorflow** : used for deep learning models.

## Approches & Methodologies
    
-  Performed a quick overview about the dataset like the dataset shape , data types, and detected any missing values. Therefore, if there were any problems associated
   within the dataset.

   * within this dataset, there are some data issues need to be fixed :

     1. remove unnecessary variables in the dataset that don't add any value to the analysis like identifiers, or the features that don't have much variation and don't add any value for the neural
        networks prediction power, in which help us to reduce the dimensionality too.
     2. handling the missing/null values by using **SimpleImputer**.

- Performed exploratory data analysis (EDA) : univariate analysis, and bivariate analysis that will help us to describe and summarize the dataset characteristics, identify associations
  between the variables, or reveal insights, and recognize any kind of patterns within the dataset that will help us to identify which features are important predictors that
  uniquely distinguish between customers behavior, in which is crucial to the prediction power of neural networks.
  
- Preprocessing data before modeling by : define the predictors and the target, encoding categorical variables by creating dummy variables, and finally split our dataset into train-test datasets
  **(train dataset size : 70% - test dataset size : 30%)**.

  ### *Before we move to the modeling and the evaluation, there are some important points to consider:*

From our earlier analysis we found that there are 79% of customers who don't churn, while 20% of them churned which suggests an unbalanced dataset and it could be challenging to classify the correct labels 
since we have a few instances of the churned customers.

  1. So, due to an unbalanced data, we may suffer some bias in the final results, that happens when the model performing very well in predicting those customers who will not churn since it has more
     observations for this particular class **(Major Class)**, and it 's very poor in predicting those customers who will churn **(Minor Class)** which are the targeted customers for this problem.
  2. It's crucial to the bank which wants to predict the customers who really intend to leave and don't waste the time on those who will not.In which the bank may end up spending time and resources on the wrong customers who don't plan to leave while they lose customers who are actually planning to leave. 
     In fact, prediction errors are costly and bad when the rate is really high. In which, the bank either ends up wasting the resources or losing the customers.
     When the model is not accurate in predicting the customers who don't planning to leave as a customers planning to means it will end up by consuming the resources based on wrong prediction , or when the model failed to predict those customers who intend to leave as they don't intend to,  the bank will end up losing their customers without even knowing.
     
## Modeling

In order to solve this binary classification problem, we built two different neural networks architectures by user-defined functions:

- The first classifier version :
  
     - It has Sequential().
     - Three dense layers with using **Relu** as activation function in the hidden layer, and **Sigmoid** for the output layer.
     - It used **BatchNormalization()** to normalize the layers' inputs by re-centering and re-scaling (we can use Dropout() to avoid overfitting, but for this particular case we will use 
       BatchNormalization() as it works better with the data we have).
     - It used callbacks in keras to help us in a proper training of the model.
     - It used callbacks **ReduceLROnPlateau** to reduce the learning rate when the validation loss has stopped improving.
     - t will return a history plot of Accuracy and Loss during the training epochs.
     - For this model version the batch size that will be used here is **Mini Batch Gradient Descent**.
     - **ROC-AUC Curve** will be used to select the optimal threshold.
       
- The second classifier version:
  
   It's the same as the previous classifier with slight differences:
     - The activation function for the hidden layers that will be used with this architecture is **Tanh**.
     - The callback in keras that will be used here is **Early Stopping** which will interrupt the training process when the validation loss is no longer improving after exactly 15 epochs.
 
 ### *However, the number of the epochs, and the optimizer is different for each model we will train.*

 **This a screenshot of the model acurracy during training for the first classifier version with Adadelta optimizer**
 
![accr_train](https://github.com/hayasalman/Artificial-Neural-Network-ANN/assets/71796909/3466e605-5d63-465d-95b2-07010186ee81)

**This a screenshot of the model loss during training for the first classifier version with Adadelta optimizer**

![loss_train](https://github.com/hayasalman/Artificial-Neural-Network-ANN/assets/71796909/5ef2ef18-1c01-4217-a776-601d1a4f9cf1)

**This a screenshot of the ROC-AUC Curve for the first classifier version with Adadelta optimizer**

![roc_](https://github.com/hayasalman/Artificial-Neural-Network-ANN/assets/71796909/64416f12-1686-4e52-9b74-4f24c65ae2d1)

## Performance Evaluation & Draw Conclusions

In terms of determining which of the models have the best performance results and will be used as a final solution.
And as it's classification task we will be using these following metrics that embedded within sklearn package or we can use user-defined function to compute the results: 

1. **Classification report** : accuray, precision, recall, F-score.
   - **Sensitivity** is the metric that evaluates a model's ability to predict true positives of each available category.
   - **Specificity** is the metric that evaluates a model's ability to predict true negatives of each available category.

And the best model is the model which has a high percentage  of (precision and recall) for both categories.

2. **Confuestion matrix**
    - The false positive and false negative rates must be minimized as possible.

**Conclusion : after comparison between these multiple models performance results**

- The first classifier version with **Adadelta optimizer and 2800 epochs** has the best performance, and it's our best solution so far. Of course, it took more epochs to use during training the model, and it was slow to learn as we observed, but it did a satisfactory job compared to all the other models.
  
- Otherwise , the other models were either biased toward the upper-hand class, unstable during the training process, or have poor performance.
  
- Also, we can go further and try to tune these model hyperparameters that will lead us to identify the best learning rate and batch size that will help to get out the local optima.

  **This a screenshot of the performance results of the final solution - The first classifier version with Adadelta optimizer after 2800 epochs**

  
  ![class_rept](https://github.com/hayasalman/Artificial-Neural-Network-ANN/assets/71796909/c74ef280-a7c4-46c9-933f-e017797e7b8a)

##  Business Insights & Recommendations 

- **Insight 1** : we found that the chances of the churn are higher among those customers who complained before than those who didn't.Therefore, the bank must pay more attention
   toward those customers as they are the most likely to churn, and the fact the bank has this information, in which a complaint is a vital factor to retain the customers or not , it's advisable 
   to develop a strategic plan to minimize the loss of their customers.

- **Insight 2** : moreover, we observed the churn rate is **30%** in germany alone and approximately **25%** are the female customers. Obviously the churn more among customers either from german or females, in which we may say, if the customer is german women who did complain before has a higher chances to churn than any other customers , and the bank potentially will end up by losing this customer, whereas it may suffers to keep up their market share in german if they keeps losing those customers to the competitors for a different reasons.
  
## References

[Bank's Customers Churn Prediction Project File](https://github.com/hayasalman/Artificial-Neural-Network-ANN/blob/main/Customers_Churn_ANN_Notebook_.ipynb)

## Suggestions

- For further improvements, we could use hypertuning to determine the best parameters that can give us the ANN model with the lowest number of epochs needed to be train and the best batch size, and learning rate to stabilize our learning process.
  
- Since we deal with a binary classification problem, we can use a different classification algorithms to predict outcomes like: **naive bayes classifier, support vector machine (SVM), decision trees classifier (DT), or random forest classifier**, and compare its performance to the neural network model.


   
 


    

  
