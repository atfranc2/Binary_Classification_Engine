#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


### Define a function that calculates the average squared error of the classifications

def avg_sqrd_error( true_class, pred_probs, classes = 2 ):
    
    df = pd.DataFrame( true_class )
    
    #Define the actual probability of the event being a 1
    df[ 'True_P1' ] = np.where( true_class == 1, 1, 0 )
    
    #Define the actual probability of the event being a 0
    df[ 'True_P0' ] = np.where( true_class == 0, 1, 0 )
    
    #Define the predicted probability of an event being a 1
    df[ 'P1' ] = pred_probs
    
    #Define the predicted probability of an event being a 0
    df[ 'P0' ] = 1 - df[ 'P1' ]
    
    #Calculate the squared error of the prediction versus the real probability of the event
    df['SE'] = ( ( df['True_P1'] - df['P1'] )**2 ) + ( ( df['True_P0'] - df['P0'] )**2 )
    
    #Calculates the average squared error of the dataframe
    ASE = df['SE'].sum() / ( len(df) * classes )
    
    #Return the average squared error (ASE) and standard deviation of the error
    return [ ASE , df['SE'].std() ]
    


# In[ ]:


### Define a function that calculates the average squared error of the classifications

def fit_metrics( true_class, pred_probs, classes = 2 ):
    
    df = pd.DataFrame( true_class )
    
    #Define the actual probability of the event being a 1
    df[ 'True_P1' ] = np.where( true_class == 1, 1, 0 )
    
    #Define the actual probability of the event being a 0
    df[ 'True_P0' ] = np.where( true_class == 0, 1, 0 )
    
    #Define the predicted probability of an event being a 1
    df[ 'P_1' ] = pred_probs
    
    #Define the predicted probability of an event being a 0
    df[ 'P_0' ] = 1 - df[ 'P_1' ]
    
    #Calculate the squared error of the prediction versus the real probability of the event
    df['SE'] = ( ( df['True_P1'] - df['P_1'] )**2 ) + ( ( df['True_P0'] - df['P_0'] )**2 )
        
    #Return the average squared error (ASE) and standard deviation of the error
    return df


# In[ ]:


def ridge_probs( fitted_model_object, test_data ):
    
    decisions = fitted_model_object.decision_function( test_data )
    
    probabilities = ( np.exp( decisions ) / ( 1 + np.exp( decisions ) ) )
    
    return probabilities


# In[ ]:


def auc( cutoff_stat_df ):
    
    areas = [ ]
    
    for i in range (1, len( cutoff_stat_df ) ):
        
        w = abs(cutoff_stat_df['FPR'].iloc[ i ] - cutoff_stat_df['FPR'].iloc[ i - 1 ])
        
        a = cutoff_stat_df['TPR'].iloc[ i ]
        
        b = cutoff_stat_df['TPR'].iloc[ i - 1 ]
        
        area = ((a+b)*w) / 2
        
        areas.append( area )
        
    return sum(areas)


# In[ ]:


def classify( pred_proba_1, prop_cutoff = 0.5 ): 
    
    pred_proba_1 = pd.DataFrame( pred_proba_1, columns = ['Predicted Probability'] )
    
    pred_proba_1['Predicted Class'] = np.where( pred_proba_1['Predicted Probability'] >= prop_cutoff, 1, 0 )
    
    return pred_proba_1


# In[ ]:


def get_lift( classify_df, true_class, orig_target_vec , plot = False ):
    
    prop_1 = len( orig_target_vec[ orig_target_vec == 1 ] ) / len( orig_target_vec )
    
    depths = [ i/100 for i in range( 1, 100, 1 ) ]
    
    classify_df['True Class'] = true_class
    
    classify_df = classify_df.sort_values( by = 'Predicted Probability' , ascending = False )
    
    lift_vec = [ ]
    
    for depth in depths: 
        
        resp_set = classify_df.iloc[ 0: int( len( classify_df )*depth )]
        
        model_resp_rate = len( resp_set[ resp_set[ 'True Class' ] == 1 ] ) / len( resp_set )
        
        null_resp_rate = ( len( resp_set ) * prop_1 ) / len( resp_set )
        
        lift = model_resp_rate / null_resp_rate
        
        lift_vec.append( lift )
    
    if plot == True: 
        plt.plot( list(map( lambda x: x*100, depths)) , lift_vec )
        plt.title( 'Lift of the Model at Depth' )
        plt.xlabel( 'Depth (%)' )
        plt.ylabel( 'Lift' )
        plt.grid()
        
    return lift_vec


# In[ ]:


def pred_probs( P_1 ): 
    
    P_0 = 1 - P_1
    
    pred_proba = pd.DataFrame([P_1, P_0], index = ['P_1', 'P_0']).transpose()
    
    return pred_proba


# In[ ]:


def model_summary(pred_y, true_y, sqrd_err, levels = 2):
    
    perf = pd.crosstab(true_y, pred_y)
    
    if perf.shape == (2,1):
    
        if perf.columns.values[0] == 0:
            
            TN  = perf.iloc[0,0]
            
            FP  = 0
            
            FN  = perf.iloc[1,0]
            
            TP  = 0
            
        if perf.columns.values[0] == 1:
            
            TN  = 0 
            
            FP  = perf.iloc[0,0]
            
            FN  = 0
            
            TP  = perf.iloc[1,0]
            
    else: 
            
        TN  = perf.iloc[0,0]
        FP  = perf.iloc[0,1]
        FN  = perf.iloc[1,0]
        TP  = perf.iloc[1,1]
            
    


    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    FNR = FN / (FN + TP)
    
    try: 
    
        PPV = TP / (TP + FP)
    
    except: 
        
        PPV = 0
    
    accuracy = (TP + TN) / (TP+TN+FP+FN)
    
    ase = sqrd_err.sum() / (levels * len(sqrd_err))
    
    fit_metrics = [TN, FP, FN ,TP ,TPR, TNR, FPR, FNR, ase, PPV, accuracy]
    
    fit_cols = ['TN', 'FP', 'FN' ,'TP' ,'TPR', 'TNR', 'FPR', 'FNR', 'ASE', 'PPV', 'Accuracy']
    
    return [ fit_cols, fit_metrics ]


# In[ ]:


def cutoff_stats( fit_summary_stats, actual_y = 'target1', sqrd_error = 'SE' ):
    
    pred_probs = fit_summary_stats
    
    ks_cutoffs = np.linspace(0.01,0.99,101)

    col_names = ['ks', 'TN', 'FP', 'FN' ,'TP' ,'TPR', 'TNR', 'FPR', 'FNR', 'ASE', 'PPV', 'Accuracy']

    ks_summary = []

    for i in ks_cutoffs:

        pred_probs[ 'Pred_Y' ] = np.where(pred_probs[ 'P_1' ] >= i , 1, 0 )

        class_perfrom = model_summary(pred_probs['Pred_Y'].values, pred_probs[ actual_y ].values, pred_probs[ sqrd_error ])[1]

        class_perfrom.insert(0, i)

        ks_summary.append( class_perfrom )

    results = pd.DataFrame(ks_summary, columns = col_names)

    results['Youden(J)'] = results['TPR'] + results['TNR'] - 1

    max_J_prob = results[results['Youden(J)'] == results['Youden(J)'].max()]['ks'].values[0]

    results['F1_score'] = 2*((results['PPV'] * results['TPR']) / (results['PPV'] + results['TPR']))

    max_F1_prob = results[results['F1_score'] == results['F1_score'].max()]['ks'].values[0]
    
    return results


# In[ ]:


def get_cutoffs( cutoff_data, original_data, fit_summary_stats, target_name ):
    
    Youden_cuttoff = cutoff_data[cutoff_data[ 'Youden(J)' ] == cutoff_data[ 'Youden(J)' ].max()]['ks'].values[0]
    
    F1_cutoff = cutoff_data[cutoff_data[ 'F1_score' ] == cutoff_data[ 'F1_score' ].max()]['ks'].values[0]
    
    halfway_cutoff = cutoff_data['ks'].median()
    
    ### Calculate the KS-Cutoff
    
    sorted_preds = fit_summary_stats.sort_values(by = 'P_1')

    proportions = [[],[]]

    true_prop1 = len(original_data[original_data[target_name] == 1 ]) / len(original_data)

    total1 = len(sorted_preds[sorted_preds[target_name] == 1])

    total0 = len(sorted_preds[sorted_preds[target_name] == 0])

    for i in sorted_preds['P_1']:

        prop1 = len(sorted_preds[(sorted_preds['P_1'] <= i) & (sorted_preds[target_name] == 1)]) / total1

        prop0 = len(sorted_preds[(sorted_preds['P_1'] <= i) & (sorted_preds[target_name] == 0)]) / total0
        
        proportions[0].append(prop0)
        
        proportions[1].append(prop1)
        
    ks_curve = pd.DataFrame(proportions, index = ['Prop0', 'Prop1']).transpose()
    
    ks_curve['P_1'] = sorted_preds['P_1']
    
    ks_curve['Prop_Diff'] = ks_curve['Prop0'] - ks_curve['Prop1']
    
    ks_cutoff = ks_curve['Prop_Diff'].max()
    
    cutoff_names = ['Youden', 'F1_Score', '50_50', 'KS']
    
    cutoff_values = [Youden_cuttoff, F1_cutoff, halfway_cutoff, ks_cutoff]
    
    return [ cutoff_names, cutoff_values ]
    


# In[ ]:


def plotting_engine( cutoff_output, cutoffs ):
    
    ks = cutoff_output['ks']
    
    TPR = cutoff_output['TPR']
    
    TNR = cutoff_output['TNR']
    
    FPR = cutoff_output['FPR']
    
    FNR = cutoff_output['FNR']
    
    PPV = cutoff_output['PPV']
    
    Jouden = cutoffs[1][0]
    
    F1_Score = cutoffs[1][1]
    
    halfway = cutoffs[1][2]
    
    KS_cutoff = cutoffs[1][3]
    
    Accuracy = cutoff_output['Accuracy']
    
    #Plot the ROC curve
    plt.figure(figsize=(8,8))
    plt.plot(FPR, TPR)
    plt.plot(FPR, FPR, ls='--')
    plt.title('ROC Plot')
    plt.xlabel('1 - Specificity [FPR]')
    plt.ylabel('Sensitivity [TPR]')
    plt.show()
    
    #Plot true negative and true positive rate
    plt.figure(figsize=(8,8))
    plt.plot(ks, TNR)
    plt.plot(ks, TPR)
    plt.axvline(Jouden, ls=':', color = 'r')
    plt.axvline(F1_Score, ls=':', color = 'b')
    plt.axvline(halfway, ls=':', color = 'y')
    plt.axvline(KS_cutoff, ls=':', color = 'g')
    plt.legend(['TNR', 'TPR', 'Jouden_Cutoff', 'F1_Cutoff', '50_50', 'KS_Cutoff'])
    plt.title('True Negative and True Positive Rates Across Cutoff Probabilities')
    plt.xlabel('Probability Cutoff')
    plt.ylabel('Truth Rate')
    plt.show()
    
    #Plot false positive and false negative rates 
    plt.figure(figsize=(8,8))
    plt.plot(ks, FNR)
    plt.plot(ks, FPR)
    plt.axvline(Jouden, ls=':', color = 'r')
    plt.axvline(F1_Score, ls=':', color = 'b')
    plt.axvline(halfway, ls=':', color = 'y')
    plt.axvline(KS_cutoff, ls=':', color = 'g')
    plt.legend(['FNR', 'FPR', 'Jouden_Cutoff', 'F1_Cutoff', '50_50', 'KS_Cutoff'])
    plt.title('False Negative and False Positive Rates Across Cutoff Probabilities')
    plt.xlabel('Probability Cutoff')
    plt.ylabel('False Rate')
    plt.show()
    
    #Plot PPV 
    plt.figure(figsize=(8,8))
    plt.plot(ks, PPV)
    plt.axvline(Jouden, ls=':', color = 'r')
    plt.axvline(F1_Score, ls=':', color = 'b')
    plt.axvline(halfway, ls=':', color = 'y')
    plt.axvline(KS_cutoff, ls=':', color = 'g')
    plt.legend(['PPV', 'Jouden_Cutoff', 'F1_Cutoff', '50_50', 'KS_Cutoff'])
    plt.title('Positive Predictive Value (PPV) Rates Across Cutoff Probabilities')
    plt.xlabel('Probability Cutoff')
    plt.ylabel('Positive Predictive Value (PPV)')
    plt.show()
    
    #Plot Accuracy
    plt.figure(figsize=(8,8))
    plt.plot(ks, Accuracy)
    plt.axvline(Jouden, ls=':', color = 'r')
    plt.axvline(F1_Score, ls=':', color = 'b')
    plt.axvline(halfway, ls=':', color = 'y')
    plt.axvline(KS_cutoff, ls=':', color = 'g')
    plt.legend(['Accuracy', 'Jouden_Cutoff', 'F1_Cutoff', '50_50', 'KS_Cutoff'])
    plt.title('Accuracy Across Cutoff Probabilities')
    plt.xlabel('Probability Cutoff')
    plt.ylabel('Accuracy')
    plt.show()
    
    
#plotting_engine( cutoff_results, cutoff_set)


# In[ ]:


def binary_classification_engine( P_1, actual_y, original_data, target_name = 'target1'):
    
    fit_summary = fit_metrics( true_class = actual_y, pred_probs = P_1, classes = 2 )
    
    cutoff_stat_results = cutoff_stats( fit_summary_stats = fit_summary, actual_y = target_name, sqrd_error = 'SE' )
    
    cutoff_set = get_cutoffs( cutoff_data = cutoff_stat_results, 
                              
                             original_data = original_data, 
                             
                             fit_summary_stats = fit_summary, 
                            
                             target_name = target_name )
    
    plotting_engine( cutoff_output = cutoff_stat_results, cutoffs = cutoff_set )
    
    return cutoff_stat_results
    
    

