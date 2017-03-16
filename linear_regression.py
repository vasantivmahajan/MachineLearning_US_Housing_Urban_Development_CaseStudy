import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#pip install scikit-neuralnetwork 
#conda update scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit,StratifiedKFold


"""dataset=pd.DataFrame("historical_data1_Q12005.csv")
print(dataset.head(10))

X_train = dataset.data[:-20]
X_test  = dataset.data[-20:]
Y_train = dataset.target[:-20]
Y_test  = dataset.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)
print(regr.coef_)
np.mean((regr.predict(X_test)-Y_test)**2)
regr.score(X_test, Y_test) """

#using label encoder to convert categorical columns into numeric values
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df

def compute_coefficients(dataframe):
        #using statsmodel estimate the model coefficients for predictions
        credit_score = smf.ols(formula='ORIGINAL_INTEREST_RATE~CREDIT_SCORE', data=dataframe).fit()
        combined_loan_to_value = smf.ols(formula='ORIGINAL_INTEREST_RATE~ORGINAL_COMBINED_LOAN_TO_VALUE', data=dataframe).fit()
        Original_debt_to_income_ratio=smf.ols(formula='-~ORIGINAL_DEBT_TO_INCOME_RATIO', data=dataframe).fit()
        Original_loan_to_value=smf.ols(formula='ORIGINAL_INTEREST_RATE~ORIGINAL_LOAN_TO_VALUE', data=dataframe).fit()
        Unpaid_principal_balance=smf.ols(formula='ORIGINAL_INTEREST_RATE~ORIGINAL_UPB', data=dataframe).fit()
        Property_state=smf.ols(formula='ORIGINAL_INTEREST_RATE~PROPERTY_STATE', data=dataframe).fit()
        Property_type=smf.ols(formula='ORIGINAL_INTEREST_RATE~PROPERTY_TYPE', data=dataframe).fit()
        Loan_purpose=smf.ols(formula='ORIGINAL_INTEREST_RATE~LOAN_PURPOSE', data=dataframe).fit()
        Original_Loan_Term=smf.ols(formula='ORIGINAL_INTEREST_RATE~ORIGINAL_LOAN_TERM', data=dataframe).fit()
        Number_of_borrowers=smf.ols(formula='ORIGINAL_INTEREST_RATE~NUMBER_OF_BORROWERS', data=dataframe).fit()
        Seller_name=smf.ols(formula='ORIGINAL_INTEREST_RATE~SELLER_NAME', data=dataframe).fit()
        Servicer_name=smf.ols(formula='ORIGINAL_INTEREST_RATE~SERVICER_NAME', data=dataframe).fit()

        
        # print the coefficients
        print(credit_score.params)
        print(combined_loan_to_value.params)
        print(Original_debt_to_income_ratio.params)
        print(Unpaid_principal_balance.params)
        print(Original_loan_to_value.params)
        print(Property_state.params)
        print(Property_type.params)
        print(Loan_purpose.params)
        print(Original_Loan_Term.params)
        print(Number_of_borrowers.params)
        print(Seller_name.params)
        print(Servicer_name.params)
        

#load the dataframe
dataframe = pd.read_csv("2005Q1.csv")
dataframe_quarter = pd.read_csv("2005Q2.csv")
#feature selection based on coefficient values
feature_cols_dataframe=dataframe[["PROPERTY_STATE",'PROPERTY_TYPE',"LOAN_PURPOSE",'SERVICER_NAME','ORIGINAL_INTEREST_RATE']]
feature_cols=["PROPERTY_STATE",'PROPERTY_TYPE',"LOAN_PURPOSE",'SERVICER_NAME','ORIGINAL_INTEREST_RATE']

#transforming categorical/string data into numeric data for the algorithm
#used custom encoding instead of one hot encoding for memory efficiency
transformed_df=dummyEncode(feature_cols_dataframe)
print(transformed_df.head(5))
transformed_df_next_quarter=dummyEncode(feature_cols_dataframe)
print(transformed_df_next_quarter.head(5))
#X = transformed_df
#y = transformed_df.Original_interest_rate

#split the data for training and testing 
# X_train = transformed_df.data[:-20]
# X_test  = transformed_df.data[-20:]
# Y_train = transformed_df.Original_interest_rate.target[:-20]
# Y_test  = transformed_df.Original_interest_rate.target[-20:]
X_train=transformed_df
Y_train=transformed_df.ORIGINAL_INTEREST_RATE

X_train=transformed_df_next_quarter
Y_train=transformed_df_next_quarter.ORIGINAL_INTEREST_RATE


#linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train)
print ("Intercept is ",linear_reg.intercept_)
print("Coefficient is ",linear_reg.coef_)
#print(lm.predict([18,3,0,4]))
print("Training score is ",linear_reg.score(X_train, Y_train))

np.mean((linear_reg.predict(X_test)-Y_test)**2)
print("Testing score is ",linear_reg.score(X_test, Y_test))

#neural network
# scaler = StandardScaler()
# # Fit only to the training data
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# neural_network_reg=MLPClassifier(hidden_layer_sizes=(20,10,20))
# neural_network_reg.fit(X_train,y_train)
# predictions = neural_network_reg.predict(X_test)
# print("Intercept is ",neural_network_reg.intercept_)
# print("Coefficient is ",neural_network_reg.coef_)
# print("Training score is ",neural_network_reg.score(X_train, Y_train))
# print("Testing score is ",neural_network_reg.score(X_test, Y_test))


"""
# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 6, sharey=True)
dataframe.plot(kind='scatter', x='Credit_score', y='Original_interest_rate', ax=axs[0], figsize=(16, 8))
dataframe.plot(kind='scatter', x='Original_combined_loan-to-value', y='Original_interest_rate', ax=axs[1])
dataframe.plot(kind='scatter', x='Original_debt_to_income_ratio', y='Original_interest_rate', ax=axs[2])

dataframe.plot(kind='scatter', x='Unpaid_principal_balance', y='Original_interest_rate', ax=axs[3])
dataframe.plot(kind='scatter', x='Original-loan-to-value', y='Original_interest_rate', ax=axs[4])
dataframe.plot(kind='scatter', x='Property_state', y='Original_interest_rate', ax=axs[5])"""