import os
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import clf_reg_models as m
import datetime as dt

def save_results(t, pred_arr, y_test, report_path):
    f = open(report_path, "a+")
    cm = confusion_matrix(y_test, pred_arr, [0, 1])
    # print(cm)
    f.write(str(t) + "," + str(cm[0][0]) + "," + str(cm[0][1]) + "," + str(cm[1][0]) + "," + str(cm[1][1]) + ",")
    f.write(str(precision_score(y_test, pred_arr)) + "," + str(recall_score(y_test, pred_arr)) + ",")
    try:
        fpr = float(cm[0][1]) / float((cm[0][1] + cm[1][0]))
    except:
        fpr = 0
    f.write(str(mean_squared_error(y_test, pred_arr)) + ",")
    f.write(str(fpr) + "," + str(accuracy_score(pred_arr, y_test) * 100) + "\n")
    f.close()


def vary_threshold(pred_arr, y_test, report_path, threshold_range):
    for t in threshold_range:
        bin_pred_arr = []
        for i in range(0, len(pred_arr)):
            if pred_arr[i] < t:
                bin_pred_arr.append(0)
            else:
                bin_pred_arr.append(1)
        save_results(t, bin_pred_arr, y_test, report_path)


def get_ml_models(model_list):
    model_arr = []
    model_names = []
    predicts_probability = []
    for model in model_list:
        model_arr.append(m.ml_models[model][0])
        model_names.append(m.ml_models[model][1])
        predicts_probability.append(m.ml_models[model][2])
    return model_arr, model_names, predicts_probability


""" Dump models for multiple training sets in a folder, to create ensemble """
def dump_models(input_lookup_folder, model_list, save_folder):
    for file in os.listdir(input_lookup_folder):
        print("Processing " + file)
        base_name = file.split(".")[0]
        df = pd.read_csv(input_lookup_folder + "/" + file, header=0, dtype="float32")
        df = append_ctr(df)
        df = binarize(df, "CTR", 0)
        features, labels = get_train_set(df)
        models, model_names = get_ml_models(model_list)
        for i in range(0, len(models)):
            models[i].fit(features, labels)
            joblib.dump(models[i], save_folder + "/" + model_names[i] + "_" + base_name + ".pkl", compress=True)
            print("   Dumped " + file + ".pkl")

""" Dump model for a single training set """
def dump_model(model, save_folder, name):
    joblib.dump(model, save_folder + "/" + name + ".pkl", compress=True)


def load_model(path):
    return joblib.load(path)

def get_ensemble_pred_arr(model_lookup_folder, features):
    result_df = pd.DataFrame()
    for file in os.listdir(model_lookup_folder):
        print("Loading " + file + "...")
        model = joblib.load(model_lookup_folder + "/" + file)
        print("   Model " + file + " is Predicting...")
        pred_arr = pd.Series(model.predict(features))
        result_df = result_df.append(pred_arr, ignore_index=True)
    pred_arr = result_df.mean()
    return pred_arr


def get_train_set(df, drop_fields_initial):
    for field in drop_fields_initial:
        df.drop(field, axis=1, inplace=True)
    df.dropna(inplace=True)
    dataset = df.as_matrix()
    features = dataset[:, :-1]
    labels = dataset[:, features.shape[1]:].ravel()
    return features, labels


def append_ctr(df):
    df['CTR'] = df['click'] / df['impression']
    df['CTR'] *= 100
    df.drop('impression', axis=1, inplace=True)
    df.drop('click', axis=1, inplace=True)
    # return df


def binarize(df, column_name, threshold):
    df[column_name] = df[column_name].apply(lambda x: 1 if float(x) > threshold else 0)
    # return df


def write_report_header(header, report_path):
    f = open(report_path, "a+")
    f.write("\n" + header + "\n")
    f.close()


def gridSearch(model, params):
    return GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', n_jobs=-1, iid=False, cv=3)


def dump_gs_results(gsearch_model, base_model_name, dataset_name):
    f = open("GridSearchParams.csv", "a+")
    f.write(base_model_name + ",Dataset," + dataset_name + "\n")
    best_parameters = gsearch_model.best_estimator_.get_params()
    for param_name in sorted(best_parameters.keys()):
        f.write(param_name + "," + str(best_parameters[param_name]) + "\n")
    f.close()

def save_predictions_against_clid(reportPath, clids, pred_arr):
    f = open(reportPath + ".csv", "w+")
    for i in range(0, len(clids)):
        f.write(str(clids[i]) + "," + str(pred_arr[i]) + "\n")
    f.close()

def save_coef(path, model):
    f = open(path + "coef.csv", "a+")
    coef = model.coef_
    for list_ in coef:
            for value in list_:
                f.write(str(value)+',')
    f.write('\n')
    f.close()

def getLogger(name, dir=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if dir is None:
        dir = "./logs/"
    if not logger.handlers:
        hdlr = logging.FileHandler('%s/%s.%s.log'%(dir,name,str(dt.date.today())))
        formatter = logging.Formatter('%(asctime)s %(process)d %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
    logger.setLevel(level)
    return logger
