# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 19:13:22 2022

@author: Rakesh
"""

###################Problem 1######################
import pandas as pd
import numpy as np

#loading dataset#
calories = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_SLR/calories_consumed.csv')

calories.describe()
##lets makes changes of names in columns#
cols= {'Weight gained (grams)':'Weight_gained', 'Calories Consumed':'Calories_Consumed'}
calories.rename(cols,axis=1, inplace=True)
calories.columns 
##Graphical representation#
import matplotlib.pyplot as plt
plt.bar(height=calories['Weight_gained'], x=np.arange(1,15,1))
plt.hist(calories['Weight_gained'])
plt.boxplot(calories['Weight_gained'])

plt.bar(height=calories['Calories_Consumed'], x=np.arange(1,15,1))
plt.hist(calories['Calories_Consumed'])
plt.boxplot(calories['Calories_Consumed'])

##scatter plot#
plt.scatter(x=calories['Weight_gained'], y=calories['Calories_Consumed'], color= 'red')

##correlation#
np.corrcoef(calories['Weight_gained'],calories['Calories_Consumed'])

#covariance#
cov_output = np.cov(calories['Weight_gained'],calories['Calories_Consumed'])

##Importing stats model #
import statsmodels.formula.api as smf

##simple linear regression#
model = smf.ols('Calories_Consumed~Weight_gained' ,data=calories).fit()
model.summary()

pred1= model.predict(pd.DataFrame(calories['Weight_gained']))

##regression line##
plt.scatter(x=calories['Weight_gained'], y=calories['Calories_Consumed'], color= 'red')
plt.plot(calories.Weight_gained, pred1,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res1 = calories.Calories_Consumed-pred1
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1

#Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at
plt.scatter(x=np.log(calories['Weight_gained']), y=calories['Calories_Consumed'], color='blue')
np.corrcoef(np.log(calories.Weight_gained),calories.Calories_Consumed)

model2 = smf.ols('Calories_Consumed~np.log(Weight_gained)' ,data=calories).fit()
model.summary()

pred2= model2.predict(pd.DataFrame(calories['Weight_gained']))    


##regression line##
plt.scatter(np.log(calories.Weight_gained ), calories.Calories_Consumed)
plt.plot(np.log(calories.Weight_gained), pred2,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res2 = calories.Calories_Consumed-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)

plt.scatter(x=calories['Weight_gained'], y=np.log(calories['Calories_Consumed']), color= 'brown')
np.corrcoef(calories.Weight_gained),np.log(calories.Calories_Consumed)

model3 = smf.ols('np.log(Calories_Consumed)~Weight_gained' ,data=calories).fit()
model3.summary()

pred3= model3.predict(pd.DataFrame(calories['Weight_gained']))
pred3_at= np.exp(pred3)    

##regression line##
plt.scatter(calories.Weight_gained , np.log(calories.Calories_Consumed))
plt.plot(calories.Weight_gained, pred3,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res3 = calories.Calories_Consumed-pred3
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)
model4 = smf.ols('np.log(Calories_Consumed)~Weight_gained+ I(Weight_gained*Weight_gained)',data= calories).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(calories))
pred4_at= np.exp(pred4)
pred4_at

##Regression line#
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=calories.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(calories.Weight_gained,np.log(calories.Calories_Consumed))
plt.plot(X, pred4, color='red')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res4 = calories.Calories_Consumed-pred4
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4

##Choosing Model using RMSE#
data = {'MODEL':pd.Series(['SLR', 'Log model', 'EXP model', 'Poly model']), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse=pd.DataFrame(data)
table_rmse

##model -best one#
from sklearn.model_selection import train_test_split

train,test = train_test_split(calories, test_size=0.2)

finalmodel = smf.ols('Calories_Consumed~Weight_gained',data= train).fit()
finalmodel.summary()

##prediction test data#
test_pred = finalmodel.predict(pd.DataFrame(test))

##model evaluation on Test data#
test_res = test.Calories_Consumed - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

##predict on train data#
train_pred = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.Calories_Consumed - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse



#####################Problem 2###########################
import pandas as pd
import numpy as np
##loading data#
delivery = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_SLR/delivery_time.csv')

delivery.describe()

cols= {'Delivery Time':'Delivery_Time', 'Sorting Time':'Sorting_Time'}
delivery.rename(cols,axis=1 , inplace= True)
delivery.columns
###Plotting#
import matplotlib.pyplot as plt
plt.bar(height=delivery['Delivery_Time'], x=np.arange(1,22,1))
plt.hist(delivery['Delivery_Time'])
plt.boxplot(delivery['Delivery_Time'])

plt.bar(height=delivery['Sorting_Time'], x=np.arange(1,22,1))
plt.hist(delivery['Sorting_Time'])
plt.boxplot(delivery['Sorting_Time'])

##scatter plot#
plt.scatter(x=delivery['Delivery_Time'], y=delivery['Sorting_Time'], color= 'red')

##correlation#
np.corrcoef(delivery['Delivery_Time'],delivery['Sorting_Time'])

#covariance#
cov_output = np.cov(delivery['Delivery_Time'],delivery['Sorting_Time'])[0,1]
cov_output

##Importing stats model #
import statsmodels.formula.api as smf

##simple linear regression#
model = smf.ols('Delivery_Time~Sorting_Time' ,data=delivery).fit()
model.summary()

pred1= model.predict(pd.DataFrame(delivery['Sorting_Time']))

##regression line##
plt.scatter(x=delivery['Delivery_Time'], y=delivery['Sorting_Time'], color= "blue")
plt.plot(delivery.Delivery_Time, pred1,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res1 = delivery.Delivery_Time-pred1
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x=np.log(delivery['Sorting_Time']), y=delivery['Delivery_Time'], color='brown')
np.corrcoef(np.log(delivery.Sorting_Time),delivery.Delivery_Time)

model2 = smf.ols('Delivery_Time~np.log(Sorting_Time)' ,data=delivery).fit()
model2.summary()


pred2= model2.predict(pd.DataFrame(delivery['Sorting_Time']))

##regression line##
plt.scatter(np.log(delivery.Sorting_Time), delivery.Delivery_Time)
plt.plot(np.log(delivery.Sorting_Time), pred2,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res2= delivery.Delivery_Time-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#Exponential transformation
# x = waist; y = log(at)
plt.scatter(x=delivery['Sorting_Time'], y=np.log(delivery['Delivery_Time']), color='orange')
np.corrcoef(delivery.Sorting_Time,np.log(delivery.Delivery_Time))

model3 = smf.ols('np.log(Delivery_Time)~Sorting_Time' ,data=delivery).fit()
model3.summary()

pred3= model3.predict(pd.DataFrame(delivery['Sorting_Time']))
pred_at = np.exp(pred3)

##regression line##
plt.scatter(delivery.Sorting_Time, np.log(delivery.Delivery_Time))
plt.plot(delivery.Sorting_Time, pred3,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res3= delivery.Delivery_Time-pred3
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3

#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = delivery).fit()
model4.summary()

pred4= model4.predict(pd.DataFrame(delivery))
pred4_at = np.exp(pred4)
pred4_at

##Regression line#
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=delivery.iloc[:, 1:2].values
X_poly = poly_reg.fit_transform(X)

##regression line##
plt.scatter(delivery.Sorting_Time,np.log(delivery.Delivery_Time))
plt.plot(X, pred4,color='blue')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res4= delivery.Delivery_Time-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4

##Choosing Model using RMSE#
data = {'MODEL':pd.Series(['SLR', 'Log model', 'EXP model', 'Poly model']), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse=pd.DataFrame(data)
table_rmse

##model -best one#
from sklearn.model_selection import train_test_split

train,test = train_test_split(delivery, test_size=0.2)

finalmodel = smf.ols('Delivery_Time~np.log(Sorting_Time)',data= train).fit()
finalmodel.summary()

##prediction test data#
test_pred = finalmodel.predict(pd.DataFrame(test))

##model evaluation on Test data#
test_res = test.Delivery_Time- test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

##predict on train data#
train_pred = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.Delivery_Time - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

########################Problem 3###################################
import pandas as pd
import numpy as np

##loading data#
emp = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_SLR/emp_data.csv')
emp.describe()

##plotting#

import matplotlib.pyplot as plt


plt.bar(height=emp['Salary_hike'], x=np.arange(1,11,1))
plt.hist(emp['Salary_hike'])
plt.boxplot(emp['Salary_hike'])

plt.bar(height=emp['Churn_out_rate'], x=np.arange(1,11,1))
plt.hist(emp['Churn_out_rate'])
plt.boxplot(emp['Churn_out_rate'])

##scatter plot#
plt.scatter(x=emp['Salary_hike'], y=emp['Churn_out_rate'], color= 'red')

##correlation#
np.corrcoef(emp['Salary_hike'],emp['Churn_out_rate'])

#covariance#
cov_output = np.cov(emp['Salary_hike'],emp['Churn_out_rate'])[0,1]
cov_output

# Import library
import statsmodels.formula.api as smf

##simple linear regression#
model = smf.ols('Churn_out_rate~Salary_hike' ,data=emp).fit()
model.summary()

pred1= model.predict(pd.DataFrame(emp['Salary_hike']))

##regression line##
plt.scatter(x=emp['Salary_hike'], y=emp['Churn_out_rate'], color= "blue")
plt.plot(emp.Salary_hike, pred1,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res1 = emp.Churn_out_rate-pred1
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1

# Log Transformation
# x = log(waist); y = at

plt.scatter(x=np.log(emp['Salary_hike']), y=emp['Churn_out_rate'], color='brown')
np.corrcoef(np.log(emp.Salary_hike),emp.Churn_out_rate)

model2 = smf.ols('Churn_out_rate~np.log(Salary_hike)' ,data=emp).fit()
model2.summary()


pred2= model2.predict(pd.DataFrame(emp['Salary_hike']))

##regression line##
plt.scatter(np.log(emp.Salary_hike), emp.Churn_out_rate)
plt.plot(np.log(emp.Salary_hike), pred2,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res2= emp.Churn_out_rate-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)
plt.scatter(x=emp['Salary_hike'], y=np.log(emp['Churn_out_rate']), color='orange')
np.corrcoef(emp.Salary_hike,np.log(emp.Churn_out_rate))

model3 = smf.ols('np.log(Churn_out_rate)~Salary_hike' ,data=emp).fit()
model3.summary()

pred3= model3.predict(pd.DataFrame(emp['Salary_hike']))
pred_at = np.exp(pred3)

##regression line##
plt.scatter(emp.Salary_hike, np.log(emp.Churn_out_rate))
plt.plot(emp.Salary_hike, pred3,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res3= emp.Salary_hike-pred3
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3

#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = emp).fit()
model4.summary()

pred4= model4.predict(pd.DataFrame(emp))
pred4_at = np.exp(pred4)
pred4_at

##Regression line#
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=emp.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

##regression line##
plt.scatter(emp.Salary_hike,np.log(emp.Churn_out_rate))
plt.plot(X, pred4,color='blue')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res4= emp.Churn_out_rate-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4

##Choosing Model using RMSE#
data = {'MODEL':pd.Series(['SLR', 'Log model', 'EXP model', 'Poly model']), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse=pd.DataFrame(data)
table_rmse

##model -best one#
from sklearn.model_selection import train_test_split

train,test = train_test_split(emp, test_size=0.2)

finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = train).fit()
finalmodel.summary()

##prediction test data#
test_pred = finalmodel.predict(pd.DataFrame(test))

##model evaluation on Test data#
test_res = test.Churn_out_rate- test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

##predict on train data#
train_pred = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.Churn_out_rate - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

###its found that training and test value is almost same #


########################################Problem 4#############################################

import pandas as pd
import numpy as np

##loading dataset#
salary = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_SLR/Salary_Data.csv')

salary.describe()

#plotting##
import matplotlib.pyplot as plt

plt.bar(height=salary['YearsExperience'], x=np.arange(1,31,1))
plt.hist(salary['YearsExperience'])
plt.boxplot(salary['YearsExperience'])

plt.bar(height=salary['Salary'], x=np.arange(1,31,1))
plt.hist(salary['Salary'])
plt.boxplot(salary['Salary'])

##scatter plot#
plt.scatter(x=salary['YearsExperience'], y=salary['Salary'], color= 'red')

##correlation#
np.corrcoef(salary['YearsExperience'],salary['Salary'])

#covariance#
cov_output = np.cov(salary['YearsExperience'],salary['Salary'])[0,1]
cov_output

# Import library
import statsmodels.formula.api as smf

##simple linear regression#
model = smf.ols('Salary~YearsExperience' ,data=salary).fit()
model.summary()

pred1= model.predict(pd.DataFrame(salary['YearsExperience']))

##regression line##
plt.scatter(x=salary['YearsExperience'], y=salary['Salary'], color= "blue")
plt.plot(salary.YearsExperience, pred1,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res1 = salary.Salary-pred1
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1

# Log Transformation
# x = log(waist); y = at

plt.scatter(x=np.log(salary['YearsExperience']), y=salary['Salary'], color='brown')
np.corrcoef(np.log(salary.YearsExperience),salary.Salary)

model2 = smf.ols('Salary~np.log(YearsExperience)' ,data=salary).fit()
model2.summary()


pred2= model2.predict(pd.DataFrame(salary['YearsExperience']))

##regression line##
plt.scatter(np.log(salary.YearsExperience), salary.Salary)
plt.plot(np.log(salary.YearsExperience), pred2,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res2= salary.Salary-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)
plt.scatter(x=salary['YearsExperience'], y=np.log(salary['Salary']), color='orange')
np.corrcoef(salary.YearsExperience,np.log(salary.Salary))

model3 = smf.ols('np.log(Salary)~YearsExperience' ,data=salary).fit()
model3.summary()

pred3= model3.predict(pd.DataFrame(salary['YearsExperience']))
pred_at = np.exp(pred3)

##regression line##
plt.scatter(salary.YearsExperience, np.log(salary.Salary))
plt.plot(salary.YearsExperience, pred3,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res3= salary.YearsExperience-pred3
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3

#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = salary).fit()
model4.summary()

pred4= model4.predict(pd.DataFrame(salary))
pred4_at = np.exp(pred4)
pred4_at

##Regression line#
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=salary.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

##regression line##
plt.scatter(salary.YearsExperience,np.log(salary.Salary))
plt.plot(X, pred4,color='blue')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res4= salary.Salary-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4

##Choosing Model using RMSE#
data = {'MODEL':pd.Series(['SLR', 'Log model', 'EXP model', 'Poly model']), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse=pd.DataFrame(data)
table_rmse

##model -best one#
from sklearn.model_selection import train_test_split

train,test = train_test_split(salary, test_size=0.2)

finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = train).fit()
finalmodel.summary()

##prediction test data#
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Salary = np.exp(test_pred)
pred_test_Salary

##model evaluation on Test data#
test_res = test.Salary- pred_test_Salary
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

##predict on train data#
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Salary = np.exp(train_pred)
pred_train_Salary

# Model Evaluation on train data
train_res = train.Salary - pred_train_Salary
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

####################################Problem 5#################################################
import pandas as pd
import numpy as np
##loading data set#

sat = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_SLR/SAT_GPA.csv')
sat.describe()

##plotting #

import matplotlib.pyplot as plt

plt.bar(height=sat['SAT_Scores'], x=np.arange(1,201,1))
plt.hist(sat['SAT_Scores'])
plt.boxplot(sat['SAT_Scores'])

plt.bar(height=sat['GPA'], x=np.arange(1,201,1))
plt.hist(sat['GPA'])
plt.boxplot(sat['GPA'])

##scatter plot#
plt.scatter(x=sat['GPA'], y=sat['SAT_Scores'], color= 'blue')

##correlation#
np.corrcoef(sat['SAT_Scores'],sat['GPA'])

#covariance#
cov_output = np.cov(sat['SAT_Scores'],sat['GPA'])[0,1]
cov_output

# Import library
import statsmodels.formula.api as smf

##simple linear regression#
model = smf.ols('SAT_Scores~GPA' ,data=sat).fit()
model.summary()

pred1= model.predict(pd.DataFrame(sat['GPA']))

##regression line##
plt.scatter(x=sat['GPA'], y=sat['SAT_Scores'], color= "grey")
plt.plot(sat.GPA, pred1,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res1 = sat.SAT_Scores-pred1
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
# Log Transformation
# x = log(waist); y = at

plt.scatter(x=np.log(sat['GPA']), y=sat['SAT_Scores'], color='brown')
np.corrcoef(np.log(sat.GPA),sat.SAT_Scores)

model2 = smf.ols('SAT_Scores~np.log(GPA)' ,data=sat).fit()
model2.summary()


pred2= model2.predict(pd.DataFrame(sat['GPA']))

##regression line##
plt.scatter(np.log(sat.GPA), sat.SAT_Scores)
plt.plot(np.log(sat.GPA), pred2,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res2= sat.SAT_Scores-pred2
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2

#Exponential transformation
# x = waist; y = log(at)
plt.scatter(x=sat['GPA'], y=np.log(sat['SAT_Scores']), color='orange')
np.corrcoef(sat.GPA,np.log(sat.SAT_Scores))

model3 = smf.ols('np.log(SAT_Scores)~GPA' ,data=sat).fit()
model3.summary()

pred3= model3.predict(pd.DataFrame(sat['GPA']))
pred3_at = np.exp(pred3)

##regression line##
plt.scatter(sat.GPA, np.log(sat.SAT_Scores))
plt.plot(sat.GPA, pred3,'r')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res3= sat.SAT_Scores-pred3_at
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3

#Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = sat).fit()
model4.summary()

pred4= model4.predict(pd.DataFrame(sat))
pred4_at = np.exp(pred4)
pred4_at

##Regression line#
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X=sat.iloc[:, 1:2].values
X_poly = poly_reg.fit_transform(X)

##regression line##
plt.scatter(sat.GPA,np.log(sat.SAT_Scores))
plt.plot(X, pred4,color='blue')
plt.legend(['Predicted line', 'Observed Data'])
plt.show()

##error calculation#
res4= sat.SAT_Scores-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4

##Choosing Model using RMSE#
data = {'MODEL':pd.Series(['SLR', 'Log model', 'EXP model', 'Poly model']), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse=pd.DataFrame(data)
table_rmse

##model -best one#
from sklearn.model_selection import train_test_split

train,test = train_test_split(sat, test_size=0.2)

finalmodel = smf.ols('SAT_Scores ~ np.log(GPA)', data = train).fit()
finalmodel.summary()

##prediction test data#
test_pred = finalmodel.predict(pd.DataFrame(test))


##model evaluation on Test data#
test_res = test.SAT_Scores- test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

##predict on train data#
train_pred = finalmodel.predict(pd.DataFrame(train))


# Model Evaluation on train data
train_res = train.SAT_Scores - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

######################################################################################################













