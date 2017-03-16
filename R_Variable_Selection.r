memory.limit(size=10222500)
memory.limit()

#dir_name <- "Historical_Loan_Data/historical_data1_Q12005/historical_data1_Q12005.csv"
dir_name <- "Data/Sample/sample_2016/sample_orig_2016.csv"

myData <- read.csv(dir_name, row.names=NULL)

# can be removed when nans are replaced
myData <- na.omit(myData)

colnames(myData) <- c("CREDIT_SCORE",
                      "FIRST_PAYMENT_DATE",
                      "FIRST_TIME_HOMEBUYER_FLAG",
                      "MATURITY_DATE",
                      "MSA",
                      "MORTAGAGE_INSURANCE_PERCENTAGE",
                      "NUMBER_OF_UNITS",
                      "OCCUPANCY_STATUS",
                      "ORGINAL_COMBINED_LOAN_TO_VALUE",
                      "ORIGINAL_DEBT_TO_INCOME_RATIO",
                      "ORIGINAL_UPB",
                      "ORIGINAL_LOAN_TO_VALUE",
                      "ORIGINAL_INTEREST_RATE",
                      "CHANNEL",
                      "PREPAYMENT_PENALTY_MORTGAGE_FLAG",
                      "PRODUCT_TYPE",
                      "PROPERTY_STATE",
                      "PROPERTY_TYPE",
                      "POSTAL_CODE",
                      "LOAN_SEQUENCE_NUMBER",
                      "LOAN_PURPOSE",
                      "ORIGINAL_LOAN_TERM",
                      "NUMBER_OF_BORROWERS",
                      "SELLER_NAME",
                      "SERVICER_NAME",
                      "SUPER_CONFORMING_FLAG"
)

# myData = head(myData)
myData
myData[,'PROPERTY_TYPE']



install.packages("MASS", dependencies=TRUE)
library(MASS)



# approach 4 all-in-one
myData

# show if/which colums have unique values > 1
lapply(myData[-1], var, na.rm = TRUE) != 0

# filter out the columns that do not have unique values > 1
myData <- Filter(function(x)(length(unique(x))>1), myData)

#myData

# variable 
min.model <- lm(ORIGINAL_INTEREST_RATE ~ 1, data=myData)
fwd.model <- step(min.model, direction = "both", scope = (~ CREDIT_SCORE+
                                                               FIRST_PAYMENT_DATE+
                                                               FIRST_TIME_HOMEBUYER_FLAG+
                                                               MATURITY_DATE+
                                                               MSA+
                                                               MORTAGAGE_INSURANCE_PERCENTAGE+
                                                               NUMBER_OF_UNITS+
                                                               OCCUPANCY_STATUS+
                                                               ORGINAL_COMBINED_LOAN_TO_VALUE+
                                                               ORIGINAL_DEBT_TO_INCOME_RATIO+
                                                               ORIGINAL_UPB+
                                                               ORIGINAL_LOAN_TO_VALUE+
                                                               ORIGINAL_INTEREST_RATE+
                                                               CHANNEL+
                                                               PREPAYMENT_PENALTY_MORTGAGE_FLAG+
                                                               # PRODUCT_TYPE+
                                                               PROPERTY_STATE+
                                                               PROPERTY_TYPE+
                                                               POSTAL_CODE+
                                                               # LOAN_SEQUENCE_NUMBER+
                                                               LOAN_PURPOSE+
                                                               ORIGINAL_LOAN_TERM+
                                                               NUMBER_OF_BORROWERS+
                                                               SELLER_NAME+
                                                               SERVICER_NAME+
                                                               SUPER_CONFORMING_FLAG))
#, scope = (~ x1 + x2 + ... xn)