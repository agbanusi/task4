'''Beginning'''
def credit():
    '''This is a function that get he prediction of credit and also outputs it's accuracy'''
    import pandas as pd
    import numpy as np
    df = pd.read_csv('C:/Users/Ademola/Desktop/dataset/credit card/crx.data')
    #clean up the messed up data
    df.replace('?', np.NAN, inplace=True)
    #check the null values
    print(df.isnull().sum())
    #fill the numeric null with the mean values
    df.fillna(df.mean(), inplace=True)
    #fill the non numeric values
    df.fillna(method='ffill', inplace=True)
    #check now the null values
    print(df.isnull().sum())
    #convert all string and non numeric codes to numeric encoding
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    for i in df:   
        df[i] = labelencoder.fit_transform(df[i])
    #scale all te dataset to achieve better accuracy
    from sklearn.preprocessing import MinMaxScaler
    sca = MinMaxScaler()
    sca.fit(df)
    sca_features = sca.transform(df)
    df = pd.DataFrame(sca_features, columns=df.columns[:])    
    df.drop(['00202', 'f'], axis=1, inplace=True)
    #split into X and y 
    X = df.drop('+', axis=1)
    y = df['+']
    #split into test and train
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)
    #import your model to use
    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    pred = logmodel.predict(X_test).astype(dtype='int64')
    pred2 = labelencoder.inverse_transform(pred)
    #ensure the predicted value is in int form because of the scaling
    y_test = y_test.astype(dtype='int64')
    y_tester = labelencoder.inverse_transform(y_test) 
    #Display the results in confusion matrix, classification result and accuracy score
    print('The predicted Values', pred2)
    from sklearn.metrics import classification_report
    print('Classification Report: ', '\n', classification_report(y_tester, pred2))
    print('\n')
    from sklearn.metrics import confusion_matrix
    print('Confusion Matrix is: ', '\n', confusion_matrix(y_tester, pred2))
    print('\n')   
    from sklearn.metrics import accuracy_score
    print('Accuracy Score is:', accuracy_score(y_tester, pred2))
    print('\n')
    # Import GridSearchCV
    from sklearn.model_selection import GridSearchCV
    # Define the grid of values for tol and max_iter
    TOL = [0.01, 0.001, 0.0001]
    MAX_ITER = [100, 150, 200]
    # Create a dictionary
    Param_grid = dict(tol=TOL, max_iter=MAX_ITER)
    # Initializing GridSearchCV
    Grid_model = GridSearchCV(estimator=LogisticRegression(), param_grid=Param_grid, cv=5)
    # Calculating and summarizing the final results
    Grid_model_result = Grid_model.fit(X, y)
    BEST_SCORE, BEST_PARAMS = Grid_model_result.best_score_, Grid_model_result.best_params_
    print("Best: %f using %s" %  (BEST_SCORE, BEST_PARAMS))            
if __name__ == '__main__':
    credit()
              