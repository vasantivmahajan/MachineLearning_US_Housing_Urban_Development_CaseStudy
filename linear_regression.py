import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit,StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model


#using label encoder to convert categorical columns into numeric values
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                if feature!='ORIGINAL_INTEREST_RATE':
                    df[feature] = le.fit_transform(df[feature])
                else:
                    df[feature] = df[feature] 
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
        #ADDED BY PRANJAL 
        Postal_code=smf.ols(formula='ORIGINAL_INTEREST_RATE~POSTAL_CODE', data=dataframe).fit()
        Prepayment_penalty_mortatge_flag=smf.ols(formula='ORIGINAL_INTEREST_RATE~PREPAYMENT_PENALTY_MORTGAGE_FLAG', data=dataframe).fit()
        Channel=smf.ols(formula='ORIGINAL_INTEREST_RATE~CHANNEL', data=dataframe).fit()
        Occupancy_Status=smf.ols(formula='ORIGINAL_INTEREST_RATE~OCCUPANCY_STATUS', data=dataframe).fit()
        Number_of_units=smf.ols(formula='ORIGINAL_INTEREST_RATE~NUMBER_OF_UNITS', data=dataframe).fit()
        Mortgage_insurance_percentage=smf.ols(formula='ORIGINAL_INTEREST_RATE~MORTAGAGE_INSURANCE_PERCENTAGE', data=dataframe).fit()
        MSA=smf.ols(formula='ORIGINAL_INTEREST_RATE~MSA', data=dataframe).fit()
        #MATURITY_DATE=smf.ols(formula='ORIGINAL_INTEREST_RATE~MATURITY_DATE', data=dataframe).fit()
        #First_payment_date=smf.ols(formula='ORIGINAL_INTEREST_RATE~FIRST_PAYMENT_DATE', data=dataframe).fit()
        First_time_homebuyer_flag=smf.ols(formula='ORIGINAL_INTEREST_RATE~FIRST_TIME_HOMEBUYER_FLAG', data=dataframe).fit()
        
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

dataframe['PREPAYMENT_PENALTY_MORTGAGE_FLAG'].astype(bool)
dataframe['ORIGINAL_DEBT_TO_INCOME_RATIO']=dataframe['ORIGINAL_DEBT_TO_INCOME_RATIO'].convert_objects(convert_numeric=True)
dataframe['MORTAGAGE_INSURANCE_PERCENTAGE']=dataframe['MORTAGAGE_INSURANCE_PERCENTAGE'].convert_objects(convert_numeric=True)
dataframe['FIRST_TIME_HOMEBUYER_FLAG'].astype(bool)
dataframe = dataframe.fillna(method='ffill')
#dataframe_rate_of_interest=dataframe['ORIGINAL_INTEREST_RATE'].astype('float')

dataframe_quarter = pd.read_csv("2005Q2.csv")
dataframe_quarter['PREPAYMENT_PENALTY_MORTGAGE_FLAG'].astype(bool)
dataframe_quarter['FIRST_TIME_HOMEBUYER_FLAG'].astype(bool)
dataframe_quarter['ORIGINAL_DEBT_TO_INCOME_RATIO']=dataframe_quarter['ORIGINAL_DEBT_TO_INCOME_RATIO'].convert_objects(convert_numeric=True)
dataframe_quarter['MORTAGAGE_INSURANCE_PERCENTAGE']=dataframe_quarter['MORTAGAGE_INSURANCE_PERCENTAGE'].convert_objects(convert_numeric=True)
dataframe_quarter = dataframe_quarter.fillna(method='ffill')
#dataframe_quarter_rate_of_interest=dataframe_quarter['ORIGINAL_INTEREST_RATE'].astype('float')
#dataframe_quarter['ORIGINAL_DEBT_TO_INCOME_RATIO'].astype(float)
#feature selection based on coefficient values
feature_cols_dataframe=dataframe[["PROPERTY_STATE",'PROPERTY_TYPE',"LOAN_PURPOSE",'CREDIT_SCORE','ORGINAL_COMBINED_LOAN_TO_VALUE','ORIGINAL_DEBT_TO_INCOME_RATIO','ORIGINAL_LOAN_TO_VALUE','ORIGINAL_UPB','ORIGINAL_LOAN_TERM','NUMBER_OF_BORROWERS','SELLER_NAME','SERVICER_NAME','POSTAL_CODE','CHANNEL','OCCUPANCY_STATUS','NUMBER_OF_UNITS','MORTAGAGE_INSURANCE_PERCENTAGE','ORIGINAL_INTEREST_RATE']]

#feature_cols=["PROPERTY_STATE",'PROPERTY_TYPE',"LOAN_PURPOSE",'SERVICER_NAME','ORIGINAL_INTEREST_RATE']

#transforming categorical/string data into numeric data for the algorithm
#used custom encoding instead of one hot encoding for memory efficiency
transformed_df=dummyEncode(feature_cols_dataframe)
#print(transformed_df.head(5))
transformed_df_next_quarter=dummyEncode(feature_cols_dataframe)
#print(transformed_df_next_quarter.head(5))
#X = transformed_df
#y = transformed_df.Original_interest_rate

#split the data for training and testing 

X_train=transformed_df
transformed_df['ORIGINAL_INTEREST_RATE']=transformed_df['ORIGINAL_INTEREST_RATE'].astype(float)

Y_train=transformed_df['ORIGINAL_INTEREST_RATE']
print(len(transformed_df_next_quarter))
X_test=transformed_df_next_quarter

transformed_df_next_quarter['ORIGINAL_INTEREST_RATE']=transformed_df_next_quarter['ORIGINAL_INTEREST_RATE'].astype(float)
Y_test=transformed_df_next_quarter['ORIGINAL_INTEREST_RATE']
print(len(Y_test))




model_number = input("Choose the model that you want to run:"
                      "  A. Linear Regression "
                      "  B. Neural Network "
                      "  C. KNN"
                      "  D. Random Forest")

if model_number =='A':
    
    linear_reg = linear_model.SGDRegressor(loss='epsilon_insensitive',penalty='elasticnet')
    linear_reg.fit(X_train, Y_train)
    print ("Intercept is ",linear_reg.intercept_)
    print("Coefficient is ",linear_reg.coef_)
    #print(lm.predict([18,3,0,4]))
    #print("Training score is ",linear_reg.score(X_train, Y_train))
    y_pred=linear_reg.predict(X_test)-Y_test
    #print("Testing score is ",linear_reg.score(X_test, Y_test))

    print (" Linear regression Mean Absolute Error is ",mean_absolute_error(Y_train,y_pred)*100)
    #print("Linear regression Mean Absolute Percentage Error is",mean_absolute_percentage_error(Y_train,y_pred))
    print("Linear regression Root Mean Squared Error",np.sqrt(mean_squared_error(Y_train,y_pred)))
    print("Linear regression R2 Score of the model is ",r2_score(Y_train,y_pred))
    

elif model_number =='B':
        
        #neural network
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    neural_network_reg=MLPRegressor(hidden_layer_sizes=(20,10,20))
    neural_network_reg.fit(X_train,Y_train)
    predictions = neural_network_reg.predict(X_test)
    y_pred=predictions
    #print("Intercept is ",neural_network_reg.intercept_)
    #print("Coefficient is ",neural_network_reg.coef_)
    # print("Training score is ",neural_network_reg.score(X_train, Y_train))
    # print("Testing score is ",neural_network_reg.score(X_test, Y_test))
    print ("Neural Network Mean Absolute Error is ",mean_absolute_error(Y_train,y_pred)*100)
    #print("Neural Network Mean Absolute Percentage Error is",mean_absolute_percentage_error(Y_train,y_pred))
    print("Neural Network Root Mean Squared Error",np.sqrt(mean_squared_error(Y_train,y_pred)))
    print("Neural Network R2 Score of the model is ",r2_score(Y_train,y_pred))

elif model_number == "C":
        #KNN algorithm
        print("Starting KNN algorithm")
        for K in range(25):
                 K_value = K+1
                 knn_reg = KNeighborsRegressor(n_neighbors = K_value, weights='uniform', algorithm='auto')
                 knn_reg.fit(X_train, Y_train)
                 y_pred = knn_reg.predict(X_test)
                 print("Mean Absolute Error is ",mean_absolute_error(Y_train,y_pred),"% for K-Value:",K_value)
                 #print("Mean Absolute Percentage Error is ",mean_absolute_percentage_error(Y_train,y_pred))
                 print("Root Mean Squared Error ",np.sqrt(mean_squared_error(Y_train,y_pred)))
                 print("Accuracy of the model is ",r2_score(Y_train,y_pred))

elif model_number == "D":
        print("Staring Random forest algorithm")
        random_forest = RandomForestRegressor(n_jobs=2)
        random_forest.fit(X_train, Y_train)
        y_pred=random_forest.predict(X_test)
        #predict probability of first 10 records
        #print(random_forest.predict_proba(X_test)[0:10])
        print("Mean Absolute Error is ",mean_absolute_error(Y_train,y_pred))
        #print("Mean Absolute Percentage Error is ",mean_absolute_percentage_error(Y_train,y_pred))
        print("Root Mean Squared Error ",np.sqrt(mean_squared_error(Y_train,y_pred)))
        print("Accuracy of the model is ",r2_score(Y_train,y_pred))     
        
else:
        print("Running all four algorithms in parallel")
        

        
