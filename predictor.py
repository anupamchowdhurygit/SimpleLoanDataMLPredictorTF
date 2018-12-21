#importing pandas for the DataFrame relayed functionalities
import pandas as pd
#The classifier I have used for this predictor
from sklearn.tree import DecisionTreeClassifier
#The library used to split training and validation data from the DataFrame
from sklearn.model_selection import train_test_split
#To calculate the Mean Absolute Error between the predicted and actual validation values
from sklearn.metrics import mean_absolute_error

#Loading the data I dowloaded from Kaggel website
loan_data_path = '../SimpleLoanDataMLPredictorTF/data/Loan_payments_data.csv'
loan_data = pd.read_csv(loan_data_path)

#pre processing data
#removing all NAN values
loan_data.fillna(0, inplace=True)

#Replacing unique long string values to relevant numbers
loan_data.loan_status.replace(['PAIDOFF','COLLECTION','COLLECTION_PAIDOFF'],[0,2,1], inplace =True)
loan_data.education.replace(['High School or Below','Bechalor','college','Master or Above'],[0,2,1,3], inplace =True)
loan_data.Gender.replace(['male','female'],[0,1], inplace =True)

#Setting prediction class target as the loan_status column
y = loan_data.loan_status

#Selecting handpicked features for this simple model
feature_columns=['Principal','age','terms','education','Gender','past_due_days']
X = loan_data[feature_columns]

#Creating train test split of the data
train_X, test_X, train_y, test_y = train_test_split(X,y, random_state=1)

#print(X.head())

#Setting the model
model = DecisionTreeClassifier()

model.fit(train_X,train_y)

predictions = model.predict(test_X)

#print(predictions)
#print(test_y)
#Mean absolute error for the values predicted
val_mae = mean_absolute_error(test_y,predictions)
print(val_mae) # turns out to be 0.032 - seems to be ok for now

# Will try to build a deep learning network to try to improve this and automating the selection of feature columns.
