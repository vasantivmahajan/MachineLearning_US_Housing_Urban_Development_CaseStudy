
# coding: utf-8

# In[ ]:

# @hidden_cell
credentials_2 = {
  'auth_url':'https://identity.open.softlayer.com',
  'project':'object_storage_b6d2dc43_04df_42d5_bf42_5b6ca8af67ff',
  'project_id':'b27f8cf040e5488eb7ceb30211ba0896',
  'region':'dallas',
  'user_id':'a9b41dfb203c4cb594bc77b0188ee66b',
  'domain_id':'9d54dfb7f46d48d8930e2e61e1e7ddab',
  'domain_name':'1257643',
  'username':'member_a50999dd4ffd8b8598d78f025b918355856ca82d',
  'password':"""aBoKb-UtnE7**[0a""",
  'container':'DefaultProjectbhanushalinhuskyneuedu',
  'tenantId':'undefined',
  'filename':'2005Q1.csv'
}
from io import StringIO
import requests
import json
import pandas as pd

# @hidden_cell
# This function accesses a file in your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def get_object_storage_file_with_credentials_b6d2dc4304df42d5bf425b6ca8af67ff(container, filename):
    """This functions returns a StringIO object containing
    the file content from Bluemix Object Storage."""

    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens'])
    data = {'auth': {'identity': {'methods': ['password'],
            'password': {'user': {'name': 'member_a50999dd4ffd8b8598d78f025b918355856ca82d','domain': {'id': '9d54dfb7f46d48d8930e2e61e1e7ddab'},
            'password': 'aBoKb-UtnE7**[0a'}}}}}
    headers1 = {'Content-Type': 'application/json'}
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1)
    resp1_body = resp1.json()
    for e1 in resp1_body['token']['catalog']:
        if(e1['type']=='object-store'):
            for e2 in e1['endpoints']:
                        if(e2['interface']=='public'and e2['region']=='dallas'):
                            url2 = ''.join([e2['url'],'/', container, '/', filename])
    s_subject_token = resp1.headers['x-subject-token']
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'}
    resp2 = requests.get(url=url2, headers=headers2)
    return StringIO(resp2.text)


Performance_label ={"LOAN_SEQUENCE_NUMBER": object,
                    "MONTHLY_REPORTING_PERIOD": object,
                    "CURENT_ACTUAL_UPB": float,
                    "CURRENT_LOAN_DELINQUENCY_STATUS": object,
                    "LOAN_AGE": int,
                    "REMAINING_MONTHS_TO_LEAGL_MATURITY": float,
                    "REPURCHASE_FLAG": object,
                    "MODIFICATION_FLAG": object,
                    "ZERO_BALANCE_CODE":object,
                    "ZERO_BALANCE_EFFECTIVE_DATE": object,
                    "CURRENT_INTEREST_RATE": float,
                    "CURRENT_DEFEREED_UPB": np.dtype(object),
                    "DUE_DATE_OF_LAST_PAID_INSTALLMENT":np.dtype(object),
                    "MI_RECOVERIES":np.dtype(object),
                    "NET_SALES_PROCEEDS":np.dtype(object),
                    "NON_MI_RECOVERIES":np.dtype(object),
                    "EXPENSES":np.dtype(object),
                    "LEGAL_COSTS":np.dtype(object),
                    "MAINTAINENCE_PRESERVATION_COSTS":np.dtype(object),
                    "TAXES_AND_INSURANCE":np.dtype(object),
                    "MISC_EXPENSES":np.dtype(object),
                    "ACTUAL_LOSS_CALCULATION":np.dtype(object),
                    "MODIFICATION_COST":np.dtype(object)
}

Performance_names =["LOAN_SEQUENCE_NUMBER",
                    "MONTHLY_REPORTING_PERIOD",
                    "CURENT_ACTUAL_UPB",
                    "CURRENT_LOAN_DELINQUENCY_STATUS",
                    "LOAN_AGE",
                    "REMAINING_MONTHS_TO_LEAGL_MATURITY",
                    "REPURCHASE_FLAG",
                    "MODIFICATION_FLAG",
                    "ZERO_BALANCE_CODE",
                    "ZERO_BALANCE_EFFECTIVE_DATE",
                    "CURRENT_INTEREST_RATE",
                    "CURRENT_DEFEREED_UPB",
                    "DUE_DATE_OF_LAST_PAID_INSTALLMENT",
                    "MI_RECOVERIES",
                    "NET_SALES_PROCEEDS",
                    "NON_MI_RECOVERIES",
                    "EXPENSES",
                    "LEGAL_COSTS",
                    "MAINTAINENCE_PRESERVATION_COSTS",
                    "TAXES_AND_INSURANCE",
                    "MISC_EXPENSES",
                    "ACTUAL_LOSS_CALCULATION",
                    "MODIFICATION_COST"
]



df_data_1 = pd.read_csv(get_object_storage_file_with_credentials_b6d2dc4304df42d5bf425b6ca8af67ff('DefaultProjectbhanushalinhuskyneuedu', 'total_Q12005.csv'))
# df_data_1.head()
# df_data_1.columns
df_data_2 = pd.read_csv(get_object_storage_file_with_credentials_b6d2dc4304df42d5bf425b6ca8af67ff('DefaultProjectbhanushalinhuskyneuedu', 'total_Q22005.csv'))
df_data_2.head()


# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split, KFold, StratifiedShuffleSplit,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from ggplot import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[ ]:

# def make_df_changes(df_c):
    
    

def train_df(df_data_2):
    
    #load the dataframe
    dataframe = df_data_1

    dataframe.convert_objects(convert_numeric=True)
    dataframe['CURRENT_LOAN_DELINQUENCY_STATUS']=dataframe['CURRENT_LOAN_DELINQUENCY_STATUS'].convert_objects(convert_numeric=True)
    dataframe['REPURCHASE_FLAG'].astype(bool)
    dataframe['MODIFICATION_FLAG'].astype(bool)
    dataframe.loc[dataframe['ZERO_BALANCE_CODE'].isnull(), 'ZERO_BALANCE_CODE'] = 0
    dataframe.loc[dataframe['ACTUAL_LOSS_CALCULATION'].isnull(), 'ACTUAL_LOSS_CALCULATION'] = 0
    dataframe.loc[dataframe['EXPENSES'].isnull(), 'EXPENSES'] = 0
    dataframe.loc[dataframe['MI_RECOVERIES'].isnull(), 'MI_RECOVERIES'] = 0
    dataframe['ZERO_BALANCE_CODE']=dataframe['ZERO_BALANCE_CODE'].astype('int')
    dataframe['Deliquency'] = np.where(dataframe['CURRENT_LOAN_DELINQUENCY_STATUS'] > 0 , 1, 0)
    dataframe.loc[dataframe['NON_MI_RECOVERIES'].isnull(), 'NON_MI_RECOVERIES'] = 0
    new_val = 'N'
    dataframe['MODIFICATION_FLAG']= np.where(pd.isnull(dataframe['MODIFICATION_FLAG']) ,new_val,dataframe['MODIFICATION_FLAG'] )
    #dataframe['ORIGINAL_DEBT_TO_INCOME_RATIO']=dataframe['ORIGINAL_DEBT_TO_INCOME_RATIO'].convert_objects(convert_numeric=True)
    #dataframe['MORTAGAGE_INSURANCE_PERCENTAGE']=dataframe['MORTAGAGE_INSURANCE_PERCENTAGE'].convert_objects(convert_numeric=True)
    dataframe = dataframe.fillna(method='ffill')

    dataframe_quarter = df_data_2
    dataframe_quarter['CURRENT_LOAN_DELINQUENCY_STATUS']=dataframe_quarter['CURRENT_LOAN_DELINQUENCY_STATUS'].convert_objects(convert_numeric=True)

    dataframe_quarter['REPURCHASE_FLAG'].astype(bool)
    dataframe_quarter['MODIFICATION_FLAG'].astype(bool)
    dataframe_quarter.loc[dataframe_quarter['ZERO_BALANCE_CODE'].isnull(), 'ZERO_BALANCE_CODE'] = 0
    dataframe_quarter.loc[dataframe_quarter['ACTUAL_LOSS_CALCULATION'].isnull(), 'ACTUAL_LOSS_CALCULATION'] = 0
    dataframe_quarter.loc[dataframe_quarter['EXPENSES'].isnull(), 'EXPENSES'] = 0
    dataframe_quarter.loc[dataframe_quarter['MI_RECOVERIES'].isnull(), 'MI_RECOVERIES'] = 0
    dataframe_quarter.loc[dataframe_quarter['NON_MI_RECOVERIES'].isnull(), 'NON_MI_RECOVERIES'] = 0
    dataframe_quarter['ZERO_BALANCE_CODE']=dataframe_quarter['ZERO_BALANCE_CODE'].astype('int')
    dataframe_quarter['Deliquency'] = np.where(dataframe_quarter['CURRENT_LOAN_DELINQUENCY_STATUS'] > 0 , 1,0)
    new_val = 'N'
    dataframe_quarter['MODIFICATION_FLAG']= np.where(pd.isnull(dataframe_quarter['MODIFICATION_FLAG']) ,new_val,dataframe_quarter['MODIFICATION_FLAG'] )
    #dataframe_quarter['ORIGINAL_DEBT_TO_INCOME_RATIO']=dataframe_quarter['ORIGINAL_DEBT_TO_INCOME_RATIO'].convert_objects(convert_numeric=True)
    #dataframe_quarter['MORTAGAGE_INSURANCE_PERCENTAGE']=dataframe_quarter['MORTAGAGE_INSURANCE_PERCENTAGE'].convert_objects(convert_numeric=True)
    #feature selection based on coefficient values
    ######
    dataframe_quarter = dataframe_quarter.fillna(method='ffill')
    #feature_cols_dataframe=dataframe[["PROPERTY_STATE",'PROPERTY_TYPE',"LOAN_PURPOSE",'CREDIT_SCORE','ORGINAL_COMBINED_LOAN_TO_VALUE','ORIGINAL_DEBT_TO_INCOME_RATIO','ORIGINAL_LOAN_TO_VALUE','ORIGINAL_UPB','ORIGINAL_LOAN_TERM','NUMBER_OF_BORROWERS','SELLER_NAME','SERVICER_NAME','POSTAL_CODE','CHANNEL','OCCUPANCY_STATUS','NUMBER_OF_UNITS','MORTAGAGE_INSURANCE_PERCENTAGE','ORIGINAL_INTEREST_RATE','Deliquency']]
    feature_cols_dataframe=dataframe[[
     'MONTHLY_REPORTING_PERIOD',
     #'CURENT_ACTUAL_UPB',
     'LOAN_AGE',
     'REMAINING_MONTHS_TO_LEAGL_MATURITY',
     #'REPURCHASE_FLAG',
     #'MODIFICATION_FLAG',
     'ZERO_BALANCE_CODE',
     'ZERO_BALANCE_EFFECTIVE_DATE',
     'CURRENT_INTEREST_RATE',
     #'CURRENT_DEFEREED_UPB',
     #'DUE_DATE_OF_LAST_PAID_INSTALLMENT',
     #'MI_RECOVERIES',
     #'NET_SALES_PROCEEDS',
     #'NON_MI_RECOVERIES',
     #'EXPENSES',
     'ACTUAL_LOSS_CALCULATION',
     'Deliquency']]                                                                                                                        

    #transforming categorical/string data into numeric data for the algorithm
    #used custom encoding instead of one hot encoding for memory efficiency
    transformed_df=dummyEncode(feature_cols_dataframe)
#     print(transformed_df.head(5))
    transformed_df_next_quarter=dummyEncode(feature_cols_dataframe)
#     print(transformed_df_next_quarter.head(5))

    #fetch training and test data
    Y_train=transformed_df.Deliquency

    X_train=transformed_df


    Y_test=transformed_df_next_quarter.Deliquency
    transformed_df.drop('Deliquency', axis=1, inplace=True)
#     print(Y_test.head())
    #transformed_df_next_quarter.drop('Deliquency', axis=1, inplace=True)
    X_test=transformed_df_next_quarter
    
    cm = random_forest(X_train,Y_train,X_test,Y_test)
    
    return cm
    



def random_forest(X_train,Y_train,X_test,Y_test):
     #Random Forest
#     print("Random Forest")    
    rf = RandomForestClassifier(n_jobs=2)
    rf.fit(X_train, Y_train)
    preds = rf.predict_proba(X_test)[:,1]
    y_pred=rf.predict(X_test)

    cm=confusion_matrix(Y_test, y_pred)
    print("Random Forest cm ",cm) 
    print(accuracy_score(Y_test, y_pred))
    
    return cm
    
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                if feature!='Deliquency':
                    df[feature] = le.fit_transform(df[feature])
                else:
                    print('Hello')
                    df[feature] = df[feature]
            except:
                print('Error encoding '+feature)
        return df    


columns=["Quarter", "Number_of_Actual_Delinquents","Number_of_Predicted_Delinquents","Number_of_records_in_the_dataset",
        "Number_of_Delinquents_properly_classified","Number_of_non-delinquents_improperly_classified_as_delinquents"]
df_x = pd.DataFrame(columns = columns)


df_list = []

df_list.append(df_data_2) # fetching q2_merged df
count_yrs = 1999
rowCount_x =0
for df_c in df_list:
#     print(df_c.head())
#     df_c = make_df_changes(df_c)
    
    quarter = "Q2"+str(count_yrs)
    rows_in_df = df_c['CREDIT_SCORE'].count()  
    non_delinq = df_c[(df_c['CURRENT_LOAN_DELINQUENCY_STATUS']==0)]['CURRENT_LOAN_DELINQUENCY_STATUS'].count()
    delinq = df_c[(df_c['CURRENT_LOAN_DELINQUENCY_STATUS']>0)]['CURRENT_LOAN_DELINQUENCY_STATUS'].count()
    
    cm = train_df(df_c)
 
    df_x.loc[rowCount_x, 'Quarter'] = quarter
    
    df_x.loc[rowCount_x, 'Number_of_Actual_Delinquents'] = delinq
    df_x.loc[rowCount_x, 'Number_of_Predicted_Delinquents'] = cm[0,0] +cm[1,0]
    df_x.loc[rowCount_x, 'Number_of_records_in_the_dataset'] = rows_in_df 
    df_x.loc[rowCount_x, 'Number_of_Delinquents_properly_classified'] = cm[0,0]
    df_x.loc[rowCount_x, 'Number_of_non-delinquents_improperly_classified_as_delinquents'] = cm[1,0]
    

    
    
    rowCount_x+=1
    
df_x.head()


# In[ ]:

# RFE Code to extract features
model = LogisticRegression()
rfe = RFE(model, 7)
print(X_train.columns)
fit = rfe.fit(X_train, Y_train)
print("Num Features:", fit.n_features_)
print("Selected Features" , fit.support_)
print("Feature Ranking: ", fit.ranking_)


# In[ ]:

#stats model for feature extraction
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols
logit = sm.Logit(Y_train, X_train)
affair_mod = logit.fit()
print(affair_mod.summary())


# In[ ]:

Confusion Matrix [[23622   115]
 [ 1068   195]]

