import os
import numpy as np
import scipy.io

shape = []

def MI_Preprocessing (path, Model):
    
    files = os.listdir(path)
    
    LHI_complete = []
    RHI_complete = []
    
    LH_Train = np.zeros((9,67500,6), dtype = np.double)
    RH_Train = np.zeros((9,67500,6), dtype = np.double)
    LH_Test = np.zeros((9,22500,6), dtype = np.double)
    RH_Test = np.zeros((9,22500,6), dtype = np.double)
    
    for file in range (len(files)):
        
        # Loading .mat file
        mat_data = scipy.io.loadmat(path + files[file])
        
        # Access the 'data' key
        data = mat_data['data']
        
        # Accessing the structured array within 'data'
        data_1_1 = data[0, 0]
        data_1_2 = data[0, 1]
        
        # Accessing individual fields within the structured array
        #for data (0,0)
        X_field = data_1_1['X']
        trial_field = data_1_1['trial']
        y_field = data_1_1['y']
        artifacts_field = data_1_1['artifacts']
        fs_field = data_1_1['fs']
        
        #for data (0,1)
        X_field_1 = data_1_2['X']
        trial_field_1 = data_1_2['trial']
        y_field_1 = data_1_2['y']
        artifacts_field_1 = data_1_2['artifacts']
        fs_field_1 = data_1_2['fs']
        
        # Converting fields to regular NumPy arrays
        #for data (0,0)
        X = X_field.item()
        trial = trial_field.item()
        y = y_field.item()
        artifacts = artifacts_field.item()
        fs = fs_field.item()
        
        #for data (0,1)
        X_1 = X_field_1.item()
        trial_1 = trial_field_1.item()
        y_1 = y_field_1.item()
        artifacts_1 = artifacts_field_1.item()
        fs_1 = fs_field_1.item()
        
        # Find indices where the array is equal to 1 and where it is equal to 2
        #for data(0,0)
        left_indices = np.where(y == 1)[0]
        right_indices = np.where(y == 2)[0]
        
        print('left_indices1 ' +str(np.shape(left_indices)))
        print('right_indices1 ' +str(np.shape(right_indices)))
        #for data(0,1)
        left_indices_1 = np.where(y_1 == 1)[0]
        right_indices_1 = np.where(y_1 == 2)[0]
        
        print('left_indices2 ' +str(np.shape(left_indices_1)))
        print('right_indices2 ' +str(np.shape(right_indices_1)))
        
        for i in range (len(left_indices)):
            
            L_start_sample = trial[left_indices[i]][0] + 1000 #MI start after 4 seconds of fixation 
            L_end_sample = L_start_sample + 750 #MI lasts for 3 seconds @250 Hz
            
            LHI_temp = X[L_start_sample : L_end_sample]
            
                         
            R_start_sample = trial[right_indices[i]][0] + 1000 #MI start after 4 seconds of fixation
            R_end_sample = R_start_sample + 750 #MI lasts for 3 seconds @250 Hz
            
            RHI_temp = X[R_start_sample: R_end_sample]
            
            #For integrating 2 sessions subjects
            if (i==0):
                LHI = LHI_temp
                RHI = RHI_temp
                
                shape.append(len(LHI_temp))
            else:
                LHI = np.vstack((LHI, LHI_temp))
                RHI = np.vstack((RHI, RHI_temp))
                
                shape.append(len(LHI_temp))
              
        #Appending second session into LHI and RHI
        for j in range (len(left_indices_1)):
            
            L1_start_sample = trial_1[left_indices_1[j]][0] + 1000 #MI start after 4 seconds of fixation 
            L1_end_sample = L1_start_sample + 750 #MI lasts for 3 seconds @250 Hz
            
            LHI_temp_1 = X_1[L1_start_sample: L1_end_sample]
            
            R1_start_sample = trial_1[right_indices_1[j]][0] + 1000 #MI start after 4 seconds of fixation
            R1_end_sample = R1_start_sample + 750 #MI lasts for 3 seconds @250 Hz
            
            RHI_temp_1 = X_1[R1_start_sample: R1_end_sample]
              
            LHI = np.vstack((LHI, LHI_temp_1))
            RHI = np.vstack((RHI, RHI_temp_1))
            
            shape.append(len(LHI_temp_1))
        
        if (Model == 1):  
            #Model 1
            Model_1_LHI_train = LHI[(0*750):(90*750), :] # 1 to 90 Epochs for train
            Model_1_RHI_train = RHI[(0*750):(90*750), :] # 1 to 90 Epochs for train
    
            Model_1_LHI_test = LHI[(90*750):(120*750), :] # 91 to 120 Epochs for test
            Model_1_RHI_test = RHI[(90*750):(120*750), :] # 91 to 120 Epochs for test   
            
            LH_Train[file] = Model_1_LHI_train
            RH_Train[file] = Model_1_RHI_train
            LH_Test[file] = Model_1_LHI_test
            RH_Test[file] = Model_1_RHI_test
                           
        elif (Model == 2):
            #Model 2
            Model_2_LHI_train = LHI[(30*750):(120*750), :] # 31 to 120 Epochs for train
            Model_2_RHI_train = RHI[(30*750):(120*750), :] # 31 to 120 Epochs for train
    
            Model_2_LHI_test = LHI[(0*750):(30*750), :] # 1 to 30 Epochs for test
            Model_2_RHI_test = RHI[(0*750):(30*750), :] # 1 to 30 Epochs for test   
            
            LH_Train[file] = Model_2_LHI_train
            RH_Train[file] = Model_2_RHI_train
            LH_Test[file] = Model_2_LHI_test
            RH_Test[file] = Model_2_RHI_test

        elif (Model == 3):
            #Model 3
            Model_3_LHI_train = np.vstack((LHI[(0*750):(30*750), :], LHI[(60*750):(120*750), :] )) # 1 to 30 and 61 to 120 Epochs for train
            Model_3_RHI_train = np.vstack((RHI[(0*750):(30*750), :], RHI[(60*750):(120*750), :] )) # 1 to 30 and 61 to 120 Epochs for train
    
            Model_3_LHI_test = LHI[(30*750):(60*750), :] # 31 to 60 Epochs for test
            Model_3_RHI_test = RHI[(30*750):(60*750), :] # 31 to 60 Epochs for test      
            
            LH_Train[file] = Model_3_LHI_train
            RH_Train[file] = Model_3_RHI_train
            LH_Test[file] = Model_3_LHI_test
            RH_Test[file] = Model_3_RHI_test                
                
        elif (Model == 4):
            #Model 4
            Model_4_LHI_train = np.vstack((LHI[(0*750):(60*750), :], LHI[(90*750):(120*750), :] )) # 1 to 60 and 91 to 120 Epochs for train
            Model_4_RHI_train = np.vstack((RHI[(0*750):(60*750), :], RHI[(90*750):(120*750), :] )) # 1 to 60 and 91 to 120 Epochs for train
    
            Model_4_LHI_test = LHI[(60*750):(90*750), :] # 61 to 90 Epochs for test
            Model_4_RHI_test = RHI[(60*750):(90*750), :] # 61 to 90 Epochs for test 

            LH_Train[file] = Model_4_LHI_train
            RH_Train[file] = Model_4_RHI_train
            LH_Test[file] = Model_4_LHI_test
            RH_Test[file] = Model_4_RHI_test

    LH_Train_reshaped = np.transpose(LH_Train, (1, 2, 0))
    RH_Train_reshaped = np.transpose(RH_Train, (1, 2, 0))
    LH_Test_reshaped = np.transpose(LH_Test, (1, 2, 0))
    RH_Test_reshaped = np.transpose(RH_Test, (1, 2, 0))

    return LH_Train_reshaped, RH_Train_reshaped, LH_Test_reshaped, RH_Test_reshaped

#EOG artifact rejection using regression analysis for each subject separately
def EOG_Correction (HI):
    
    sample, Ch, subj = np.shape(HI)
    
    for sub in range (subj):
        
        HI_EEG = HI[:,0:3,sub]    
        HI_EOG = HI[:,3:6,sub]    
        
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(HI_EEG, HI_EOG, test_size=1/3, random_state=0)
        
        # Create a linear regression model
        regression_model = LinearRegression()
        
        # Train the model on the training data
        regression_model.fit(X_train, y_train)
        
        #r2_score(y_test, regression_model.predict(X_test))
        
        test_eog_predictions = regression_model.predict(X_test)
        test_cleaned_eeg = X_test - test_eog_predictions
        
        train_eog_predictions = regression_model.predict(X_train)
        train_cleaned_eeg = X_train - train_eog_predictions
        
        Corrected_EEG = np.vstack((train_cleaned_eeg, test_cleaned_eeg)) 
        
        Final_Temp_EEG = np.expand_dims(Corrected_EEG, axis=-1)
        
        Final_Temp_EEG = np.transpose(Final_Temp_EEG, (2, 0, 1))
        
        if sub == 0:           
            Final_EEG = Final_Temp_EEG
            
        else:         
            Final_EEG = np.vstack((Final_EEG, Final_Temp_EEG))
            
    return r2_score(y_test, regression_model.predict(X_test)), np.transpose(Final_EEG, (1, 2, 0))


