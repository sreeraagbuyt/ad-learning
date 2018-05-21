import xgboost as xgb
import pandas as pd


BASE_PATH = "/home/buyt-app/ad_learning"
filename = "/data/pca/gender_pred_pca.csv"
xgb_model = "/models/gender_xgb.model"
probability_threshold = 0.54 # Decide using P-R values from training report
chunksize = 1000000
female_class = 1


bst = xgb.Booster()
bst.load_model(BASE_PATH + xgb_model)
df_pointer = pd.read_csv(BASE_PATH + filename, header=0, chunksize=chunksize, error_bad_lines=False)


for idx,df in enumerate(df_pointer):
    client_ids_test =  df.client_id
    X = df.drop(['client_id'], axis=1)
    dtest = xgb.DMatrix(X) 
    y_pred = bst.predict(dtest)
    y_pred_bin = [1 if item > probability_threshold else 0 for item in y_pred]
    y_pred_gender = ['F' if item == female_class else 'M' for item in y_pred_bin]
    
    print("Pred done")
    print("Saving predictions")
    
    output = pd.DataFrame({'client_id':client_ids_test, 'gender_pred': y_pred_gender, 'logprob':y_pred})
    set_header = True if idx == 0 else False
    output.to_csv(BASE_PATH+'/predictions/prod_gender_logistic_%s_pred.csv'%(idx), index = False, header=set_header)
