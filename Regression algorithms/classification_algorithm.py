import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#pip install scikit-neuralnetwork 
#conda update scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from ggplot import *
from sklearn.metrics import confusion_matrix

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

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_roc_curve(Y_test, y_pred, pos_label):
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label)
    df = pd.DataFrame(dict(fpr=fpr, tpr=true_pos_rate))
    ggplot(df, aes(x='false_pos_rate', y='true_pos_rate')) +\
    geom_line() +\
    geom_abline(linetype='dashed')

    auc = metrics.auc(fpr,tpr)
    ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +\
    geom_area(alpha=0.2) +\
    geom_line(aes(y='tpr')) +\
    ggtitle("ROC Curve w/ AUC=%s" % str(auc))

def calculate_confusion_matrix(y_true, y_pred, value_list):
    confusion_matrix(y_true, y_pred, labels=value_list)
    
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
#feature_cols=["PROPERTY_STATE",'PROPERTY_TYPE',"LOAN_PURPOSE",'SERVICER_NAME','ORIGINAL_INTEREST_RATE']

#transforming categorical/string data into numeric data for the algorithm
#used custom encoding instead of one hot encoding for memory efficiency
transformed_df=dummyEncode(feature_cols_dataframe)
print(transformed_df.head(5))
transformed_df_next_quarter=dummyEncode(feature_cols_dataframe)
print(transformed_df_next_quarter.head(5))

#fetch training and test data
X_train=transformed_df
Y_train=transformed_df.ORIGINAL_INTEREST_RATE

X_test=transformed_df_next_quarter
Y_test=transformed_df_next_quarter.ORIGINAL_INTEREST_RATE

model_number = input("Choose the model that you want to run:"
                      "  A. Logistic Regression "
                      "  B. Neural Network "
                      "  C. SVN"
                      "  D. Random Forest"
                      " "
                     )
value_list=["Delinquincy Yes", "Delinquincy No"]
if model_number == "A":
    #logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    print ("Intercept is ",log_reg.intercept_)
    print("Coefficient is ",log_reg.coef_)
    #print(lm.predict([18,3,0,4]))
    #print("Training score is ",linear_reg.score(X_train, Y_train))
    y_pred=log_reg.predict(X_test)-Y_test
    #print("Testing score is ",linear_reg.score(X_test, Y_test))

    #calculate ROC curve
    calculate_roc_curve(Y_test, y_pred,2)

    #calculate Confusion Matrix
    calculate_confusion_matrix(Y_test, y_pred, value_list)
    
    
elif model_number == "B":
        
    #neural network
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    neural_network_class=MLPClassifier(hidden_layer_sizes=(20,10,20))
    neural_network_class.fit(X_train,Y_train)
    predictions = neural_network_class.predict(X_test)
    y_pred=predictions

    #calculate ROC curve
    calculate_roc_curve(Y_test, y_pred,2)


elif model_number == "C":
        #KNN algorithm
        print("Starting KNN algorithm")
        for K in range(25):
                 K_value = K+1
                 knn_class = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
                 knn_class.fit(X_train, Y_train)
                 y_pred = knn_class.predict(X_test)
                 #calculate ROC curve
                 calculate_roc_curve(Y_test, y_pred,2)
        

elif model_number == "D":
        print("Staring Support Vector Machine")
        clf = SVC()
        clf.fit(X_train, Y_train)
        clf.predict(X_test)
        #calculate ROC curve
        calculate_roc_curve(Y_test, y_pred,2)     
        
else:
        print("Running all four algorithms in parallel")
        
