memory.limit(size=10222500)
memory.limit()



install.packages("MASS", dependencies=TRUE)
library(MASS)

#dir_name <- "Historical_Loan_Data/historical_data1_Q12005/historical_data1_Q12005.csv"
dir_name <- "Data/New_total_Q12005.csv"

myData <- read.csv(dir_name, row.names=NULL)

# can be removed when nans are replaced
# myData <- na.omit(myData)

# colnames(myData) <- c("CREDIT_SCORE",
#                       "FIRST_PAYMENT_DATE",
#                       "FIRST_TIME_HOMEBUYER_FLAG",
#                       "MATURITY_DATE",
#                       "MSA",
#                       "MORTAGAGE_INSURANCE_PERCENTAGE",
#                       "NUMBER_OF_UNITS",
#                       "OCCUPANCY_STATUS",
#                       "ORGINAL_COMBINED_LOAN_TO_VALUE",
#                       "ORIGINAL_DEBT_TO_INCOME_RATIO",
#                       "ORIGINAL_UPB",
#                       "ORIGINAL_LOAN_TO_VALUE",
#                       "ORIGINAL_INTEREST_RATE",
#                       "CHANNEL",
#                       "PREPAYMENT_PENALTY_MORTGAGE_FLAG",
#                       "PRODUCT_TYPE",
#                       "PROPERTY_STATE",
#                       "PROPERTY_TYPE",
#                       "POSTAL_CODE",
#                       "LOAN_SEQUENCE_NUMBER",
#                       "LOAN_PURPOSE",
#                       "ORIGINAL_LOAN_TERM",
#                       "NUMBER_OF_BORROWERS",
#                       "SELLER_NAME",
#                       "SERVICER_NAME",
#                       "SUPER_CONFORMING_FLAG"
# )

# myData = head(myData)

myData[,'MAINTAINENCE_PRESERVATION_COSTS']


# show if/which colums have unique values > 1
lapply(myData[-1], var, na.rm = TRUE) != 0

# filter out the columns that do not have unique values > 1
myData <- Filter(function(x)(length(unique(x))>1), myData)

#myData

# variable 
min.model <- lm(IS_DELINQUENT ~ 1, data=myData)
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
                                                               # SUPER_CONFORMING_FLAG+
                                                            #LOAN_SEQUENCE_NUMBER+
                                                            MONTHLY_REPORTING_PERIOD+
                                                            CURENT_ACTUAL_UPB+
                                                            CURRENT_LOAN_DELINQUENCY_STATUS+
                                                            LOAN_AGE+
                                                            REMAINING_MONTHS_TO_LEAGL_MATURITY+
                                                            REPURCHASE_FLAG+
                                                            MODIFICATION_FLAG+
                                                            ZERO_BALANCE_CODE+
                                                            ZERO_BALANCE_EFFECTIVE_DATE+
                                                            CURRENT_INTEREST_RATE+
                                                            CURRENT_DEFEREED_UPB+
                                                            # DUE_DATE_OF_LAST_PAID_INSTALLMENT
                                                            # MI_RECOVERIES
                                                            NET_SALES_PROCEEDS+
                                                            # NON_MI_RECOVERIES+
                                                            # EXPENSES+
                                                            # LEGAL_COSTS+
                                                            # MAINTAINENCE_PRESERVATION_COSTS+
                                                            # TAXES_AND_INSURANCE+
                                                            # MISC_EXPENSES+
                                                            # ACTUAL_LOSS_CALCULATION+
                                                            MODIFICATION_COST
                                                            ))
#, scope = (~ x1 + x2 + ... xn)