# Final Project Purwadhika
This final project is one of the requirements for graduating from Job Connector Data Science and Machine Learning Purwadhika Start-up and Coding School

<p align="center"  color="rgb(0, 90, 71)">
<h1>So, what's happening here?</h1>
</p>
<br>
<br>

<p align='justify' style="font-weight: bold;">
What's happening here is a gradient boosting framework which is called as <b>'Light GBM'</b> that uses on old well known 'Decission Tree' 
learning algorithm which comes under the more well-known umberella term 'Machine Learning'. I've trained a learning model on SBA Loans Dataset 
and now place myself as a 'Bank Officer at USA' and receive a loan application from a small business. To make the decission, I will 
use <a style="color:rgb(0, 90, 71) ;"><b>Loan Predictor</b></a> to help me to classify the risk of the loan which is a high risk or low risk Loan.
</p>

<p align="center">
<img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Dashboard/static/images/output_onlinepngtools.png?raw=true" height="100" width="180">
</p>

<p align='justify' style="font-weight: bold;">
Deeper on Loan Predictor evaluation, the accuracy of this model algorithm is ~90% and has AUC Score 0.97 which means that it is an ideal situation. 
When two curves don’t overlap at all means model has an ideal measure of separability. 
It is nearly perfectly able to distinguish between positive class and negative class. 
But, it still has a limitation that only works for the loan that has guarantee from <a href="https://www.sba.gov" style="color:rgb(0, 90, 71) ;"><b>US Small Business Administration</b></a>.
</p>

<p align="center">
<img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/SBA_logo.jpg?raw=true" height="100" width="180">
</p>

<p align='justify' style="font-weight: bold;">
The U.S. SBA was founded in 1953 on the principle of promoting and assisting small enterprises in the U.S. credit market. 
Small businesses have been a primary source of job creation in the United States; therefore, fostering small business formation and growth has social benefits by creating job opportunities and reducing unemployment. 
One way SBA assists these small business enterprises is through a loan guarantee program which is designed to encourage banks to grant loans to small businesses. 
SBA acts much like an insurance provider to reduce the risk for a bank by taking on some of the risk through guaranteeing a portion of the loan. 
In the case that a loan goes into default, SBA then covers the amount they guaranteed. 
There have been many success stories of start-ups receiving SBA loan guarantees such as FedEx and Apple Computer. 
However, there have also been stories of small businesses and/or start-ups that have defaulted on their SBA-guaranteed loans. 
  The rate of default on these loans has been a source of controversy for decades.<a href="https://doi.org/10.1080/10691898.2018.1434342">[Reference]</a></p>
  
--------------------------------------------------------------------
 <p align="center"  color="rgb(0, 90, 71)">
<h1>Step of work</h1>
</p>
<br>

### 1. Data Preprocessing and Exploratory Data Analysis (EDA)
Started with import the dataset which is SBA Loan dataset. After that, I do data cleaning to remove some unnecessary simbols, give restriction to the data that will be used based on reference and EDA, change NAICS code to industrial sector, fixed nan value, remove the outlier, etc.

### 2. Feature Engineering
At feature engineering, I add columns that maybe will be used for predicting like real estate columns. I also did one hot encoding for categorical columns like state and NAICS (Industrial Sector), drop the columns that I think it will not giving effect for modelling, check the correlation each features, etc.

### 3. Modelling
I started modelling with standardize the continues data, doing cross validation method from five algorithms which are Logistic Regression, Decission Tree, Random Forest, Light GBM, and KNN for normal data and oversampling data (using SMOTE). After found two of the best algorithms which are from normal data (without SMOTE), then I would do tuning hyperparameter for them. In the following below are the codes and results from cross validation :

```
# ------------  F1 Score from Normal Train Data (without SMOTE)
# ------------ Using F1 Score because the data imbalance, so can't trust to accuracy

algorithm = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LGBMClassifier, KNeighborsClassifier]
algo_name = ['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','LGBMClassifier','KNeighborsClassifier']
algo_F1Score = []

for item in algorithm:
  model = item()
  F1Scores = cross_val_score(model,X_train,y_train,cv=5,scoring='f1').mean()
  algo_F1Score.append(F1Scores)

df_algo = pd.DataFrame({
    'Algorithm' : algo_name,
    'F1 Score' : algo_F1Score,
})

# ------------  Plot F1 Score from Normal Data Train
plt.figure(figsize=(14,4))
sns.barplot(data=df_algo,y='Algorithm',x='F1 Score')
plt.tight_layout()
```

<p align="center"> <img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-28%20at%2012.43.10.png?raw=true" alt="" width="700" height="275"> </p>

<br>
<br>

### 4. Tuning Hyperparameter

### 5. Set Decission Maker (Threshold)

### 6. Evaluation Model

### 7. Validation Model
 
--------------------------------------------------------------------

### Home Page :
<br>
<center><img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-26%20at%2020.17.15.png?raw=true" alt="" width="950" height="550"></center>
<br>

-----------------------------------------
 
### Prediction Page :
<br>
<center><img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-26%20at%2020.17.31.png?raw=true" alt="" width="950" height="550"></center>
<br>

-----------------------------------------

### Result Page :

<br>

#### When the result is low risk loan
<br>
<center><img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-26%20at%2020.18.19.png?raw=true" alt="" width="950" height="550"></center>
<br>

#### When the result is high risk loan
<br>
<center><img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-26%20at%2020.19.02.png?raw=true" alt="" width="950" height="550"></center>
<br>

-----------------------------------------

### About Page :
<br>
<center><img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-26%20at%2020.19.21.png?raw=true" alt="" width="950" height="550"></center>
<br>

-----------------------------------------

### Data Visualization and Data Table Page :
<br>
<center><img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-26%20at%2020.19.54.png?raw=true" alt="" width="950" height="550"></center>
<br>

-----------------------------------------

### Contact Page :
<br>
<center><img src="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Pictures/Screen%20Shot%202020-05-26%20at%2020.20.07.png?raw=true" alt="" width="950" height="550"></center>
<br>


-----------------------------------------
<br>

<p align='justify' style="font-weight: bold;">
NOTES : On the dashboard, to load the final dataset, I use two choices. Using SQLAlchemy to load from SQL Databases or load manually using pandas. <a href="https://github.com/agunggnug/Final_Project_PWDK/blob/master/Dashboard/model_plots.py"> Looks here </a>
</p>

<br>
<p font-size="10px"><b>Keywords :</b> Loan, Prediction, Machine Learning, Light GBM, U.S Small Business Administration, MySQL<p>
