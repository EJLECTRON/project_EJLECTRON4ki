import pandas as pd
from sklearn.svm import SVC

data_test = pd.read_csv('Test_sumas - Test_sumas.csv')
data_train = pd.read_csv('Train_sumas - Train_sumas.csv')
data_res = pd.read_csv('Samle_Submission.csv')

data_res.drop(labels= [0,1,2,3,4,5,6,7,8,9], axis=0, inplace=True)
data_res.drop(labels= ['Id'], axis=1, inplace=True)


X_train = pd.DataFrame()
X_train['V1'] = data_train['V1']
X_train['V2'] = data_train['V2']
X_train['V3'] = data_train['V3']
X_train['V4'] = data_train['V4']
X_train['V5'] = data_train['V5']
X_train['V6'] = data_train['V6']
X_train['V7'] = data_train['V7']
X_train['V8'] = data_train['V8']
X_train['V9'] = data_train['V9']
X_train['V10'] = data_train['V10']
X_train['V11'] = data_train['V11']
X_train['V12'] = data_train['V12']
X_train['V13'] = data_train['V13']
X_train['V14'] = data_train['V14']
X_train['V15'] = data_train['V15']
X_train['V16'] = data_train['V16']
X_train['V17'] = data_train['V17']
X_train['V18'] = data_train['V18']
X_train['V19'] = data_train['V19']
X_train['V20'] = data_train['V20']
X_train['V21'] = data_train['V21']
X_train['V22'] = data_train['V22']
X_train['P1'] = data_train['P1']
X_train['P2'] = data_train['P2']
X_train['P3'] = data_train['P3']
X_train['P4'] = data_train['P4']
X_train['P5'] = data_train['P5']
X_train['P6'] = data_train['P6']
X_train['P7'] = data_train['P7']
X_train['P8'] = data_train['P8']
X_train['P9'] = data_train['P9']
X_train['P10'] = data_train['P10']
X_train['P11'] = data_train['P11']
X_train['P12'] = data_train['P12']
X_train['P13'] = data_train['P13']
X_train['P14'] = data_train['P14']
X_train['P15'] = data_train['P15']
X_train['P16'] = data_train['P16']
X_train['P17'] = data_train['P17']
X_train['P18'] = data_train['P18']
X_train['P19'] = data_train['P19']
X_train['P20'] = data_train['P20']
X_train['P21'] = data_train['P21']
X_train['P22'] = data_train['P22']
X_train['P23'] = data_train['P23']
X_train['P24'] = data_train['P24']
X_train['P25'] = data_train['P25']
X_train['P26'] = data_train['P26']
X_train['P27'] = data_train['P27']


X_test = pd.DataFrame()
X_test['V1'] = data_test['V1']
X_test['V2'] = data_test['V2']
X_test['V3'] = data_test['V3']
X_test['V4'] = data_test['V4']
X_test['V5'] = data_test['V5']
X_test['V6'] = data_test['V6']
X_test['V7'] = data_test['V7']
X_test['V8'] = data_test['V8']
X_test['V9'] = data_test['V9']
X_test['V10'] = data_test['V10']
X_test['V11'] = data_test['V11']
X_test['V12'] = data_test['V12']
X_test['V13'] = data_test['V13']
X_test['V14'] = data_test['V14']
X_test['V15'] = data_test['V15']
X_test['V16'] = data_test['V16']
X_test['V17'] = data_test['V17']
X_test['V18'] = data_test['V18']
X_test['V19'] = data_test['V19']
X_test['V20'] = data_test['V20']
X_test['V21'] = data_test['V21']
X_test['V22'] = data_test['V22']
X_test['P1'] = data_test['P1']
X_test['P2'] = data_test['P2']
X_test['P3'] = data_test['P3']
X_test['P4'] = data_test['P4']
X_test['P5'] = data_test['P5']
X_test['P6'] = data_test['P6']
X_test['P7'] = data_test['P7']
X_test['P8'] = data_test['P8']
X_test['P9'] = data_test['P9']
X_test['P10'] = data_test['P10']
X_test['P11'] = data_test['P11']
X_test['P12'] = data_test['P12']
X_test['P13'] = data_test['P13']
X_test['P14'] = data_test['P14']
X_test['P15'] = data_test['P15']
X_test['P16'] = data_test['P16']
X_test['P17'] = data_test['P17']
X_test['P18'] = data_test['P18']
X_test['P19'] = data_test['P19']
X_test['P20'] = data_test['P20']
X_test['P21'] = data_test['P21']
X_test['P22'] = data_test['P22']
X_test['P23'] = data_test['P23']
X_test['P24'] = data_test['P24']
X_test['P25'] = data_test['P25']
X_test['P26'] = data_test['P26']
X_test['P27'] = data_test['P27']

y_train = data_train['target']

test_model = SVC(kernel = 'rbf', random_state = 0)
test_model.fit(X_train, y_train)
y_test = test_model.predict(X_test)


data_result = pd.DataFrame(index=data_test['Id'])
data_result['Predicted'] = y_test

data_result.to_csv('Samle_Submission.csv')