# import packages
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from word2number import w2n
import warnings
warnings.filterwarnings('ignore')

# extract data
hiring_data = pd.read_csv("C:\\To Read\\Data_sets\\hiring.csv")

# data cleaning and preprocessing
mean_test_score = hiring_data['test_score(out of 10)'].mean()
hiring_data['test_score(out of 10)'] = hiring_data['test_score(out of 10)'].fillna(mean_test_score)
hiring_data['experience'] = hiring_data['experience'].fillna('zero')
hiring_data['experience'] = hiring_data['experience'].apply(w2n.word_to_num)

# training input and output
input_hiring_data = hiring_data.drop(columns=['salary($)'])
output_hiring_data = hiring_data['salary($)']
input_hiring_data_train, input_hiring_data_test, output_hiring_data_train, output_hiring_data_test = train_test_split(input_hiring_data, output_hiring_data, test_size=0.2)

# model
model = LinearRegression()
model.fit(input_hiring_data_train, output_hiring_data_train)
ans = model.predict(input_hiring_data_test)

# prediction
users = int(input("No of employees: "))
print()
for loop in range(users):
    experience = int(input("Enter your Experience: "))
    Test_score = int(input("Enter your test score: "))
    interview_score = int(input("Enter your interview_score: "))
    answer = model.predict([[experience, Test_score, interview_score]])
    print('Your expected salary will be:', int(answer))
    print()
