from time import time
import pandas as pd
import util as u
import datetime as dt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import Constants
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import gc

report_path = "./report/ml_report"
dump_models_folder = "./models/topic"
load_models_folder = "./models/topic"
save_predictions_folder = "./predictions/topic/full"
model_list = ["lr"]
threshold_range = map(lambda x: x / 100.0, range(0, 101, 5))
test_set_ratio = 0.2
""" Fields to be dropped from dataframe initially ( Excluding client_id, impression and click ) """
drop_fields_initial = []
logger = u.getLogger("ProcessLog")


def run(dataset_path, load_models, dump_models, vary_threshold, add_random_column):
    df = pd.read_csv(dataset_path, header=0)  
    
    """
    scaler = StandardScaler()
    for col in Constants.CONTINUOUS_COLUMNS:
        scaler.fit(df[col])
        df[col] = scaler.transform(df[col])
    """

    if add_random_column:
        df['random'] = np.random.choice(range(1, 20000), df.shape[0])
    u.append_ctr(df)
    u.binarize(df, "CTR", 0)
    features, labels = u.get_train_set(df, drop_fields_initial)
    
    """
    poly = PolynomialFeatures(2)
    features = poly.fit_transform(features[:,1:])
    names = poly.get_feature_names(Constants.FEATURES)
    """

    labels = labels.astype(int)
    if add_random_column:
        df['random'] = np.random.choice(range(1, 20000), df.shape[0])
    
    models = []
    model_names = []
    predicts_probability = []
    x_train = []
    y_train = []
    if add_random_column:
        x_test, y_test = u.get_train_set(df, [])
    else:
        x_test = features
        y_test = labels
    
    df = None
    gc.collect()

    if load_models:
        x_validation = features
        y_validation = labels
        train_set_size = str(len(x_train))
        test_set_size = str(len(x_test))
        for model in os.listdir(load_models_folder):
            models.append(u.load_model(load_models_folder + "/" + model))
            model_names.append(model.split("_")[0])
            predicts_probability.append(model.split("_")[1])
    else:
        x_train, x_validation, y_train, y_validation = train_test_split(features, labels, test_size=test_set_ratio, random_state=0)
        train_set_size = str(len(x_train))
        test_set_size = str(len(x_validation))
        models, model_names, predicts_probability = u.get_ml_models(model_list)
    
    features, labels = None, None
    input_basename = dataset_path.split("/")[-1].split(".")[0]
    report = report_path +'_'+ input_basename + ".csv"
    for i in range(0, len(models)):
        logger.info(model_names[i] +str(":")+ input_basename)
        start = time()
        if not load_models:
	    X_train, X_train_lr, y_train, y_train_lr = train_test_split(x_train,
									y_train,
									test_size=0.5)
	    rf = RandomForestClassifier(n_estimators=50, n_jobs = -1, max_depth = 8)
	    rf_enc = OneHotEncoder()
	    rf.fit(X_train[:, 1:], y_train)
	    rf_enc.fit(rf.apply(X_train[:, 1:]))
	    models[i].fit(rf_enc.transform(rf.apply(X_train_lr[:, 1:])), y_train_lr)
            logger.info("Fit complete")
	    #pred_arr = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
	    #end = time()
            #models[i].fit(x_train[:, 1:], y_train)
            if dump_models:
                u.dump_model(models[i], dump_models_folder,
                             model_names[i] + "_" + str(predicts_probability[i]) + "_" + input_basename)
        end = time()

        header = str(dt.datetime.today()) + ",Algorithm," + model_names[i] + ",Dataset," + input_basename
        header += ",Train Set Size," + train_set_size + ",Test Set Size," + test_set_size
        header += ",Time," + str((end - start) / 60) + "\n"
        u.write_report_header(header, report)

        if predicts_probability[i] == 0:
            pred_arr = models[i].predict(x_validation[:, 1:])
            if vary_threshold:
                u.vary_threshold(pred_arr, y_validation, report, threshold_range)
            else:
                u.save_results(0.5, pred_arr, y_validation, report)
            pred_arr = models[i].predict(x_test[:, 1:])
        else:
            pred_arr = models[i].predict_proba(rf_enc.transform(rf.apply(x_validation[:,1:])))[:, 1].ravel()
            #pred_arr = models[i].predict_proba(x_validation[:, 1:])[:, 1:].ravel()
            logger.info("Log loss: %s " % (log_loss(y_validation, pred_arr)))
            logger.info("ROC_AUC: %s " % (roc_auc_score(y_validation, pred_arr)))
            if vary_threshold:
                u.vary_threshold(pred_arr, y_validation, report, threshold_range)
            else:
                u.vary_threshold(pred_arr, y_validation, report, [0.5])
            X_train, X_train_lr, y_train, y_train_lr, x_train, x_validation, y_validation = None, None, None, None, None, None, None
            gc.collect()
            out = np.array_split(x_test,10)
            pred = None
            cnt = 0
            for test in out:
                cnt+=1
                pred_arr = models[i].predict_proba(rf_enc.transform(rf.apply(test[:,1:])))[:, 1].ravel()
                if cnt == 1:
                    pred = pred_arr.ravel()
                else:
                   pred = np.append(pred,pred_arr.ravel())
            #print len(pred)
            #pred_arr = models[i].predict_proba(x_test[:, 1:])[:, 1:].ravel()
        u.save_predictions_against_clid(save_predictions_folder + "/" + model_names[i] + "_" + input_basename,list(x_test[:, :1].astype(int).ravel()), list(pred))



if __name__ == "__main__":
    run("./encoded/cat_files/t.csv",False,False,False,False)
