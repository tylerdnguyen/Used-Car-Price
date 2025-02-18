# What factors drive the price of a car?

## Overview
- In this application, The dataset from Kaggle will be explored. The original dataset contained information on 3 million used cars and contains information on 426K cars to ensure speed of processing.  The goal is to understand what factors make a car more or less expensive. At the end of the analysis, clear recommendations will be given to your client -- a used car dealership -- as to what consumers value in a used car.

## CRISP-DM Framework
<center>
    <img src = images/crisp.png width = 50%/>
</center>

To frame the task, throughout our practical applications, we will refer back to a standard process in industry for data projects called CRISP-DM.  This process provides a framework for working through a data problem.  Your first step in this application will be to read through a brief overview of CRISP-DM [here](https://mo-pcco.s3.us-east-1.amazonaws.com/BH-PCMLAI/module_11/readings_starter.zip).  After reading the overview, answer the questions below.

### Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices.  In the CRISP-DM overview, we are asked to convert this business framing to a data problem definition.  Using a few sentences, reframe the task as a data task with the appropriate technical vocabulary.

**The business objective:** to understand what factors make the car less or more expensive so that we can inform our client as a used car dealership to what consumers value in a used car.
To do this:

- First, we should collect data of used car with multiple features and the prices of used car. The more data we collect, the better model will be.
- Check to see if the data we collect is valid, and make any necessary assumption.
- Analyzing the data to understand and draw conclusion which features plays important roles in influence the price of the used car.
- After determine the which features are important, we can inform our finding to our client as a guild understand what factors make a used car more or less expensive.
- Preprocessing and clean data for predict model, our goal is to find out the best model that yield lowest loss function. Evaluate the model with test set.
- Deploy the model.
- Continue evaluate the model by collecting more data to improve the quality of prediction as well as to understand more about the factors affecting the used car.

**Note:**
- This project first need data of the used car from different dealship across United States. It also requires the accuracy of the data to help us as data analyst to understand correctly what factors affect used car price. As a result, it is fair to make an assumption that resulting conclusion only valid in United States Marke.
- Collecting data and training model might be cost a lot of money; however, analytic tools/libraries are majorly free on the Internet.

### Data Understanding
After considering the business understanding, we want to get familiar with our data. Write down some steps that you would take to get to know the dataset and identify any quality issues within. Take time to get to know the dataset and explore what information it contains and how this could be used to inform your business understanding.
- Import data as dataframe using Pandas
- Get general info of the data: number of columns & rows of the data, how many feature columns, data type of each columns
- Identify if there is any missing value in the data.
- Get descriptive statistic info of each columns such as mean, std, min/max
- Check if the value of the data make sense to verify the quality of the data.
- Explore the relationship between features and the target column using visualization or correlation calculation
**Raw Data Information:**
```
RangeIndex: 426880 entries, 0 to 426879
Data columns (total 18 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            426880 non-null  int64  
 1   region        426880 non-null  object 
 2   price         426880 non-null  int64  
 3   year          425675 non-null  float64
 4   manufacturer  409234 non-null  object 
 5   model         421603 non-null  object 
 6   condition     252776 non-null  object 
 7   cylinders     249202 non-null  object 
 8   fuel          423867 non-null  object 
 9   odometer      422480 non-null  float64
 10  title_status  418638 non-null  object 
 11  transmission  424324 non-null  object 
 12  VIN           265838 non-null  object 
 13  drive         296313 non-null  object 
 14  size          120519 non-null  object 
 15  type          334022 non-null  object 
 16  paint_color   296677 non-null  object 
 17  state         426880 non-null  object 
dtypes: float64(2), int64(2), object(14)
```
![image](/images/raw_describe.png)
![image](/images/missing_frac.png)
**Quality of data:**
This data contain a significant amount of missing value (6 columns with at least 30% fraction of missing). Moreover, for numeric columns such as price and odometer, the min and max value make no sense. Therefore, the quality of the raw data is not good, which indicate the poor collection data methods. We need to handle missing data and eliminate the outliers to improve the quality of the data.

**Improve The Quality of Data**
**Drop off missing value and unnessary column:**
- For Size column, since the fraction missing is almost 70%, we have to drop the whole column.
- For region & state columns, it would be better to drop off those column since we neglect the price different between states in US.
- In addition, such columns like VIN and Id do not affect the price of the used car
- Finally, let drop all rows with missing data
**Eliminate the outliers:**
- Adjust the price range to make more sense for used car (1500 - $300000).
- Adjust the odometer range to make more sense for used car (above 10000 miles).
- Applied method interquatile range to eliminate the outilers in numeric columns

#### Data Analysis:
![image](/images/cor_matrix.png)
![image](/images/plot1.png)
![image](/images/plot2.png)

**Summary:**
- The top 3 used car manufacturers are Ford, Chevy, and Toyota. But top 3 manufacturers with highest average price are ferrari, tesla, and aston martin. As a result, we can see manufactures affect the price of the used car.
- For model, although we do see different of distribution in the dataset, but we do not see clear different in the price. We can assume different manufacturers causes price different, but within same manufacturing, model difference doesn not cause the big different in price. As a result, we can omit this model column for training model.
- Most of the used cars have clean title and gas type, with condition from like new to excellent. The Average Price does not show much different among the condition, fuel type, and title status.


### Data Preparation
After our initial exploration and fine-tuning of the business understanding, it is time to construct our final dataset prior to modeling. Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen (scaling, logarithms, normalization, etc.), and general preparation for modeling with sklearn.

- Apply James Stein Encoder to columns: manufacturer, cylinder, type, paint color
```
# Initialize the JamesSteinEncoder with the column to encode
JS_encoder = ce.JamesSteinEncoder(cols=['manufacturer','cylinders','type','paint_color']);

# Fit and transform the data
input_JSencoded = JS_encoder.fit_transform(input_df, target);

# Show the encoded training data
input_JSencoded.head()
dtypes: float64(2), int64(2), object(14)
```
- Create column transformer with OrdinalEncoder for condition, OneHotEncoder for fuel, title status, transmission, and drive
```
transformer = make_column_transformer(
        (OrdinalEncoder(), ['condition']),
        (OneHotEncoder(), ['fuel','title_status','transmission','drive']),
        remainder='passthrough'
)
```
- The rest columns passtrthrough
- Create Training and test set

### Modeling

- Build a base Model with Default Linear Regression
```
basemodel = Pipeline([
    ('transform', transformer),
    ('linreg', LinearRegression())
]);
basemodel.fit(X_train, y_train);
basemodel.score(X_test, y_test);

base_y_testPred = basemodel.predict(X_test);

base_mse = mean_squared_error(y_test, base_y_testPred);
base_score = basemodel.score(X_test,y_test);
print(f"Mean Squared Error of the base model: {base_mse}")
print(f"Score of the base model: {base_score}")
```
```
Mean Squared Error of the base model: 54992732.94674486
Score of the base model: 0.6497501639852139
```
- Use GridSearch to find optimal degree for poly feature
```
The Complexity that minimized Test Error was: 2
The Best MSE of Test Error was: 48944752.37810143
The Best Score of Test Set was: 0.688269875388525
```
- Use GridSearch with Ridge Regression + best polyfeature degree to find optimal alpha value.
```
Best alpha value: 0.001
Mean Squared Error of the best model: 40227269.159405634
Score of the best model: 0.7437917035320123
```
- Use GridSearch & Apply Sequential Feature Selection with Poly Features + Ridge Model to define the best number of feature
```
Number of features with highest accuracy score: 20
Highest accuracy score: 0.7391931735296362
```
![image](/images/SFSPlot.png)
### Evaluation
**Conclusion:**
- The features in the dataset has nonlinear characteristic since the accuracy score of the poly feature with degree of 2 is significantly higher than the base model with linear feature.
- The model is also tuned and improved by adjusting the alpha value in Ridge Model to increase the accuracy score of test set.
- After investigate number of feature using Sequential Features Selection, there was a significant increase of test set score up to 15 features. After that, the test set score flattened out.

### Deployment:
- Top 3 factors affecting the price of used car: year, odometer, cylinder types.
- Condition, transmission, and title_status do not affect much the price of the used car

### Further Studying
- Expand the detail of importance of categorical features by apply the categorial encoder then using permutaion to see specific type of cylinder/drive... that significanly affect the price of used car
- Using different categorial encoder rathan James-Stein Encoder
