# 🍫 Diabetes Prediction: Project Overview
* End to end project researching the effects personal attributes have on the diagnosis of diabetes.
* Optimised SVC using GridsearchCV and RandomizedSearchCV to reach the best model performance. 
* Deployed Model into production for wider use. See below.  


#### [Use Deployed Model](https://p7-diabetes-model.herokuapp.com/)

## Table of Contents 
*   [Resources](#resources)<br>
*   [Data Collection](#DataCollection)<br>
*   [Data Pre-processing](#DataPre-processing)<br>
*   [Data Warehousing](#DataWarehousing)<br>
*   [Exploratory data analysis](#EDA)<br>
*   [Feature Engineering](#FeatEng)<br>
*   [ML/DL Model Building](#ModelBuild)<br>
*   [Model performance](#ModelPerf)<br>
*   [Model Optimisation](#ModelOpt)<br>
*   [Model Evaluation](#ModelEval)<br>
*   [Model Productionisation](#ModelProd)<br>
*   [Deployment](#ModelDeploy)<br>
*   [Project Management (Agile/Scrum/Kanban)](#Prjmanage)<br>
*   [Project Evaluation](#PrjEval)<br>
*   [Looking Ahead](#Lookahead)<br>
*   [Questions & Contact me](#Lookahead)<br>

<a name="resources"></a>  

## Resources Used
**Python 3, PostgreSQL** 

[**Anaconda Packages:**](requirements.txt) **pandas numpy pandas_profiling ipywidgets sklearn matplotlib seaborn sqlalchemy kaggle psycopg2 ipykernel flask** <br><br>
Powershell command for installing anaconda packages used for this project  
```powershell
pip install pandas numpy pandas_profiling ipywidgets sklearn matplotlib seaborn sqlalchemy kaggle psycopg2 ipykernel flask 
```

<a name="DataCollection"></a>  

## [Data Collection](Code/P7_Code.ipynb)
Powershell command for data import using kaggle API <br>
```powershell
!kaggle datasets download -d mathchi/diabetes-data-set -p ..\Data --unzip 
```
[Data source link](https://www.kaggle.com/mathchi/diabetes-data-set)
[Data](Data/diabetes.csv)
*  Rows: 768 / Columns: 9
    *   Pregnancies                   
    *   Glucose                      
    *   BloodPressure                 
    *   SkinThickness                 
    *   Insulin                      
    *   BMI                    
    *   DiabetesPedigreeFunction    
    *   Age                          
    *   Outcome                       

<a name="DataPre-processing"></a>  

## [Data Pre-processing](Code/P7_Code.ipynb)
After I had all the data I needed, I needed to check it was ready for exploration and later modelling. I made the following changes and created the following variables:   
*   General NULL and data validity checks  
*   Formatted column headers to SQL compatibility. 
The data contained no null values and all datatypes lined up with their field description. <br>

```python
# Viewing the data types of the columns
data.dtypes

# Viewing dataset shape
data.shape

# 1st check for null values and datatype check 
data.info()
```

<br>

*   Some programming languages can be case sensitive like python and C++ for example, so using lower case letters for variable names allows for straightforward use of data in different programming languages.<br>

```python
# (SQL standard) Formatting column headers by removing potential capital letters and spaces in column headers 
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(' ','_')
```



<a name="DataWarehousing"></a>

## [Data Warehousing](Code/P7_Code.ipynb)
I warehouse all data in a Postgre database for later use and reference.

*   ETL in python to PostgreSQL Database.
*   Formatted column headers to SQL compatibility.  


```python 
# Function to warehouse data in a Postgre database 
def store_data(data,tablename):
    """
    :param data: variable, enter name of dataset you'd like to warehouse
    :param tablename: str, enter name of table for data 
    """

    # SQL table header format
    tablename = tablename.lower()
    tablename = tablename.replace(' ','_')

    # Saving cleaned data as csv
    data.to_csv(f'../Data/{tablename}_clean.csv', index=False)

    # Engine to access postgre
    engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5432/projectsdb')

    # Loads dataframe into PostgreSQL and replaces table if it exists
    data.to_sql(f'{tablename}', engine, if_exists='replace',index=False)

    # Confirmation of ETL 
    return("ETL successful, {num} rows loaded into table: {tb}.".format(num=len(data.iloc[:,0]), tb=tablename))
 
# Calling store_data function to warehouse cleaned data
store_data(data,"P7 Diabetes Prediction")
```

<a name="EDA"></a>  

## [Exploratory data analysis](Code/P7_Code.ipynb) 
I looked at the distributions of the data and the value counts for the various categorical variables that would be fed into the model. Below are a few highlights from the analysis.
*   34.9% in the dataset have diabetes.

```python
# Getting non numeric columns 
# nonnumeric_cols = data.select_dtypes(exclude=["float", 'int']).columns.tolist()

categorical_val = []
# Checking distribution of categorical fields. For loop to iterate and print value_counts for categorical values 
for i, item in enumerate(data.columns.tolist()):
    # if there is less than 9 values then continue 
    if len(data[item].value_counts()) <= 10:

        # print column name and value_counts()
        print()
        print(item)
        print(data[item].value_counts())
        categorical_val.append(item)
    # Else do nothing
    else:
        None

# IF NO OUTPUT THEN ALL COLUMNS ARE NON NUMERIC

## OUTPUT ##
outcome
0    500
1    268
Name: outcome, dtype: int64
```

```python
# Setting my custom color palette
colors = ["#32CD32","#FF0000"]
sns.set_palette(sns.color_palette(colors))
```

<img src="images/diabetes_barchart_distrib.png" />

*   There is visually a relationship between the number of pregnancies and diabetes diagnosis. 
<img src="images/independentfeatures_distrib.png" />

*   Boxplots were used to visualise features with outliers. These features will be scaled later before modelling. 
<img src="images/boxplots.png" />

*   I visualised the distribution for those with diabetes
<img src="images/histogramdistribution.png" />

*   All features are generally not strongly correlated 
<img src="images/diabetes_correlation.png" />

<!-- <a name="Dataviz"></a>  

## [Data Visualisation & Analytics](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
[View Interactive Dashboard](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
*   I created an interactive dashboard to deploy the machine learning model to benefit the business.
*   I visualised various key features and highlighted their overall correlation to a customer’s churn.  -->

<!-- <a name="Busintelli"></a>  

## Business Intelligence
On Page 2 of the interactive dashboard, I have provided the stake holders with the new customer names and the customers that are likely to churn due to their characteristics.

*   These customers can be offered subsidised deals and incentives to keep them on
*   Greater engagement with customers could keep some customers on board 
*   Providing quality customer service can also provide customers with long term value and appreciation for the business
*   The complaints team should pay particular attention to complaints from customers who are predicted to churn.
- 96% of unhappy customers don’t complain and 91% of those will simply leave and never come back? -->

<a name="FeatEng"></a>  

## [Feature Engineering](Code/P7_Code.ipynb) 
There was no need to transform the categorical variable(s) into dummy variables as they are all numeric. I also split the data into train and tests sets with a test size of 20%.
*   Using RobustScaler to scale
<!-- *   One Hot encoding to encode values -->

```python
scale_vars = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age']

# Robust scaler to address outliers 
transformer = RobustScaler().fit(data[scale_vars])
data[scale_vars] = transformer.transform(data[scale_vars])

# Using train test split to split train and test data / Stratifying so y_test can reflect y_train.Resulting in a more realistic simulation of how the model is going to perform on new data
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=23, shuffle=True, stratify=y)

# Viewing shape of train / test data
print(X_train.shape)
print(X_test.shape)
```
  

<a name="ModelBuild"></a> 

## [ML/DL Model Building](Code/P7_Code.ipynb)

I tried four different models and evaluated them using initially using accuracy_score and then MSE/RMSE. I chose MSE and RMSE because it is sensitive to outliers, punishes larger errors and is relatively easy to interpret.   

I tried four different models:
*   **KN Neighbors Classifier** 
*   **SVC** 
*   **Logistic Regression** 
*   **Random Forest Classifier**

```python
# Preparing algorithms in a list
random_state = 23
classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors = 5),
    SVC(gamma='auto',probability=True),
    RandomForestClassifier(max_features = 'sqrt', n_jobs = 1, verbose = 1)]

# Creating a df to show models and their accuracy 
log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns = log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    print("=" * 30)
    print(name)
    print('****  Results  ****')
    
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    print("\n")
    
    # Creating new df and appending results onto initial df 
    log_entry = pd.DataFrame([[name, acc * 100]], columns = log_cols)
    log = log.append(log_entry)
    
print("=" * 30)
```

<img src="images/Crossvalidation.png" />

<a name="ModelPerf"></a> 

## [Model performance](Code/P7_Code.ipynb)
The Logistic Regression model outperformed the other approaches on the test and validation sets. 
*   **Logistic Regression** : Accuracy = 77.92% 
*   **SVC** : Accuracy = 75.32% 

<a name="ModelOpt"></a> 

## [Model Optimisation](Code/P7_Code.ipynb)
In this step, I used GridsearchCV and RandomizedSearchCV to find the best parameters to optimise the performance of the model.
Using the best parameters, I improved the SVC model accuracy of SVC by **1.3%**. The Logistic Regression model however saw no increase in accuracy. 

*   **Logistic Regression** : Accuracy = 77.92% / MSE = 0.22 / RMSE =  0.47 (2dp)
*   **SVC** : Accuracy = 77.27%  / MSE = 0.23 / RMSE =  0.48 (2dp)  

<a name="ModelEval"></a> 

## [Model Evaluation](Code/P7_Code.ipynb)
*   A confusion matrix showing the accuracy score of True and False predictions achieved by the model. 

*   Logistic Regression 
<img src="images/Confusionmatrixlog.png" />

*   Support Vector Classification
<img src="images/Confusionmatrixsvc.png" />

*   Plotting the ROC Curve gave me some key insight to select the best performing model. Although Logistic regression has a higher accuracy value, I selected the SVC model as the ROC Curve shows the SVC to predict more true positives and true negative which is key for deployment and use in unseen data. 

<img src="images/ROC_Curve.png" />

<a name="ModelProd"></a> 

## [Model Productionisation](Code/P7_Code.ipynb)
*   I used the pickle library to export the model. 
```python
# Dump model into pickle file
pickle.dump(model1, open('.././svc_diabetes.pkl', 'wb'))
```  

<a name="ModelDeploy"></a> 

## [Deployment](app.py)
I built a flask REST API endpoint that was hosted on a local webserver before Heroku deployment. The API endpoint takes in request values and returns prediction of diabetes diagnosis. I also optimised and formatted the frontend using HTML and CSS. 

```python

# Creating flask app / Initiating app
app = Flask(__name__)

# Load pickle model
model = pickle.load(open("svc_diabetes.pkl", "rb"))

# Define the home page
@app.route('/')
def hello_world():
    return render_template("index.html")

# route() decorator to tell Flask what URL should trigger our function.
@app.route('/predict',methods=['POST','GET'])
# Predict method
def predict():

    # Inputs from website 
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']
    
    # Inputs into dataframe 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    

    # Finding the probability based on independent features
    prediction=model.predict_proba(row_df)

    # Formatting, selecting index 1 in predict_proba to get probability to churn or Churn == 1. Rounding to 2 d.p
    output=round(prediction[0][1], 2)

    # if output is greater than 50% proba
    if output> 0.5:

        # Converting output to string and adding % suffix
        output = str(float(output)*100)+'%'

        # Return if risk of diabetes is greater than 50%
        return render_template('result.html',pred=f'You have a higher risk of diabetes.\n\nYou have a {output} chance of having diabetes.')

    # Else if proba is less than or equal to 0.5 
    else:

        # Converting output to string and adding % suffix
        output = str(float(output)*100)+'%'

        # Return if risk of diabetes is less than or equal to 50%
        return render_template('result.html',pred=f'You have a lower risk of diabetes.\n\nYou have a {output} chance of having diabetes.')



# Only allowing file to run from this file 
if __name__ == '__main__':
    app.run(debug=True)
 
```
<a name="Prjmanage"></a> 

## [Project Management (Agile/Scrum/Kanban)](https://www.atlassian.com/software/jira)
* Resources used
    * Jira
    * Confluence
    * Trello 

<a name="PrjEval"></a> 

## [Project Evaluation]() 
*   WWW
    *   The end-to-end process
    *   Deployment and sharing of work 
*   EBI 
    *   Better project management and planning would have made this project faster
    

<a name="Lookahead"></a> 

## Looking Ahead
*   What next
*   How can a better accuracy score be achieved? 
*   Explore other data sources 

<a name="Questions"></a> 

## Questions & Contact me 
For questions, feedback, and contribution requests contact me
* ### [Click here to email me](mailto:contactmattithyahu@gmail.com) 
* ### [See more projects here](https://mattithyahudata.github.io/)


