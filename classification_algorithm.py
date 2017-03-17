
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_roc_curve(Y_test, y_pred, pos_label):
    
    fpr, tpr, _ = roc_curve(Y_test, preds)
    #Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
#     fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label)
#     df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
#     ggplot(df, aes(x=fpr, y=tpr)) +\
#     geom_line() +\
#     geom_abline(linetype='dashed')

#     auc = metrics.auc(fpr,tpr)
#     ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +\
#     geom_area(alpha=0.2) +\
#     geom_line(aes(y='tpr')) +\
#     ggtitle("ROC Curve w/ AUC=%s" % str(auc))

def calculate_confusion_matrix(y_true, y_pred):
    
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    
#using label encoder to convert categorical columns into numeric values
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                if feature!='Deliquency':
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
dataframe = df_data_4
dataframe.convert_objects(convert_numeric=True)
dataframe['CURRENT_LOAN_DELINQUENCY_STATUS']=dataframe['CURRENT_LOAN_DELINQUENCY_STATUS'].convert_objects(convert_numeric=True)
dataframe['Deliquency'] = np.where(dataframe['CURRENT_LOAN_DELINQUENCY_STATUS'] > 0 , 1, 0)
#dataframe['ORIGINAL_DEBT_TO_INCOME_RATIO']=dataframe['ORIGINAL_DEBT_TO_INCOME_RATIO'].convert_objects(convert_numeric=True)
#dataframe['MORTAGAGE_INSURANCE_PERCENTAGE']=dataframe['MORTAGAGE_INSURANCE_PERCENTAGE'].convert_objects(convert_numeric=True)
dataframe = dataframe.fillna(method='ffill')
dataframe_quarter = df_data_3
dataframe_quarter['CURRENT_LOAN_DELINQUENCY_STATUS']=dataframe_quarter['CURRENT_LOAN_DELINQUENCY_STATUS'].convert_objects(convert_numeric=True)
dataframe_quarter['Deliquency'] = np.where(dataframe_quarter['CURRENT_LOAN_DELINQUENCY_STATUS'] > 0 , 1,0)
#dataframe_quarter['ORIGINAL_DEBT_TO_INCOME_RATIO']=dataframe_quarter['ORIGINAL_DEBT_TO_INCOME_RATIO'].convert_objects(convert_numeric=True)
#dataframe_quarter['MORTAGAGE_INSURANCE_PERCENTAGE']=dataframe_quarter['MORTAGAGE_INSURANCE_PERCENTAGE'].convert_objects(convert_numeric=True)
#feature selection based on coefficient values
######
dataframe_quarter = dataframe_quarter.fillna(method='ffill')
#feature_cols_dataframe=dataframe[["PROPERTY_STATE",'PROPERTY_TYPE',"LOAN_PURPOSE",'CREDIT_SCORE','ORGINAL_COMBINED_LOAN_TO_VALUE','ORIGINAL_DEBT_TO_INCOME_RATIO','ORIGINAL_LOAN_TO_VALUE','ORIGINAL_UPB','ORIGINAL_LOAN_TERM','NUMBER_OF_BORROWERS','SELLER_NAME','SERVICER_NAME','POSTAL_CODE','CHANNEL','OCCUPANCY_STATUS','NUMBER_OF_UNITS','MORTAGAGE_INSURANCE_PERCENTAGE','ORIGINAL_INTEREST_RATE','Deliquency']]
feature_cols_dataframe=dataframe[[
 'MONTHLY_REPORTING_PERIOD',
 'CURENT_ACTUAL_UPB',
 'LOAN_AGE',
 'REMAINING_MONTHS_TO_LEAGL_MATURITY',
 'REPURCHASE_FLAG',
 'MODIFICATION_FLAG',
 'ZERO_BALANCE_CODE',
 'ZERO_BALANCE_EFFECTIVE_DATE',
 'CURRENT_INTEREST_RATE',
 'CURRENT_DEFEREED_UPB',
 'DUE_DATE_OF_LAST_PAID_INSTALLMENT',
 'MI_RECOVERIES',
 'NET_SALES_PROCEEDS',
 'NON_MI_RECOVERIES',
 'EXPENSES',
 'ACTUAL_LOSS_CALCULATION']]                                                                                                                        

#transforming categorical/string data into numeric data for the algorithm
#used custom encoding instead of one hot encoding for memory efficiency
transformed_df=dummyEncode(feature_cols_dataframe)
print(transformed_df.head(5))
transformed_df_next_quarter=dummyEncode(feature_cols_dataframe)
print(transformed_df_next_quarter.head(5))

#fetch training and test data
X_train=transformed_df
Y_train=transformed_df.Deliquency
print(Y_train.head())
X_test=transformed_df_next_quarter
Y_test=transformed_df_next_quarter.Deliquency
model_number = input("Choose the model that you want to run:"
                      "  A. Logistic Regression "
                      "  B. Neural Network "
                      "  C. SVN"
                      "  D. Random Forest"
                      " "
                     )

value_list=["Delinquincy Yes", "Delinquincy No"]
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X_train, Y_train)
print("Num Features:", fit.n_features_)
print("Selected Features" , fit.support_)
print("Feature Ranking: ", fit.ranking_)
if model_number == "A":
    #logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    print ("Intercept is ",log_reg.intercept_)
    print("Coefficient is ",log_reg.coef_)
    #print(lm.predict([18,3,0,4]))
    #print("Training score is ",linear_reg.score(X_train, Y_train))
    y_pred=log_reg.predict(X_test)
    #print("Testing score is ",linear_reg.score(X_test, Y_test))

    #calculate ROC curve
    preds = log_reg.predict_proba(X_test)[:,1]
    calculate_roc_curve(Y_test, preds,2)

    #calculate Confusion Matrix
    calculate_confusion_matrix(Y_test, y_pred)
    print(accuracy_score(Y_test, y_pred))
    
    
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
    preds = neural_network_class.predict_proba(X_test)[:,1]
    calculate_roc_curve(Y_test, preds,2)
     #calculate Confusion Matrix
    calculate_confusion_matrix(Y_test, y_pred)
    print(accuracy_score(Y_test, y_pred))


elif model_number == "C":
        #Random Forest
        rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
        rf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:,1]
        y_pred=rf.predict(X_test)
        #calculate ROC curve
        calculate_roc_curve(Y_test, y_pred,2) 
         #calculate Confusion Matrix
        
        calculate_confusion_matrix(Y_test, y_pred)
        print(accuracy_score(Y_test, y_pred))
        

elif model_number == "D":
        print("Staring Support Vector Machine")
        clf = SVC()
        clf.fit(X_train, Y_train)
        y_pred=clf.predict(X_test)
        #calculate ROC curve
        preds = clf.predict_proba(X_test)[:,1]
        calculate_roc_curve(Y_test, preds,2) 
         #calculate Confusion Matrix
        
        calculate_confusion_matrix(Y_test, y_pred)
        print(accuracy_score(Y_test, y_pred))
        
else:
        print("Running all four algorithms in parallel")
        

