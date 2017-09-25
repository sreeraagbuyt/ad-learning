from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, mean_squared_error, average_precision_score, precision_score, recall_score
from sklearn.externals import joblib
import logging
import datetime as dt
from sklearn.metrics import confusion_matrix


def save_metrics(y_test, y_pred, name):
    logger = get_logger("ModelLog")
    logger.info("Model: " + str(name))
    logger.info("Roc_Auc: "+ str(roc_auc_score(y_test, y_pred)))
    logger.info("Log Loss: "+ str(log_loss(y_test, y_pred)))
    logger.info("Avg Precision Score: "+ str(average_precision_score(y_test, y_pred)))
    logger.info("MSE: "+ str(mean_squared_error(y_test, y_pred)))
    predictions = [ round(x) for x in y_pred ]
    logger.info("Accuracy: " + str(accuracy_score(y_test, predictions)))


def write_report_header(header, report_path):
    f = open(report_path, "a+")
    f.write("\n" + header + "\n")
    f.close()


def save_results(t, pred_arr, y_test, report_path):
    f = open(report_path, "a+")
    cm = confusion_matrix(y_test, pred_arr, [0, 1])
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


def save_predictions_against_clid(reportPath, clids, pred_arr):
    f = open(reportPath + ".csv", "w+")
    for i in range(0, len(clids)):
        f.write(str(clids[i]) + "," + str(pred_arr[i]) + "\n")
    f.close()


def get_logger(name, dir=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if dir is None:
        dir = "/home/buyt-app/ad_learning/logs/"
    if not logger.handlers:
        hdlr = logging.FileHandler('%s/%s.%s.log'%(dir,name,str(dt.date.today())))
        formatter = logging.Formatter('%(asctime)s %(process)d %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
    logger.setLevel(level)
    return logger
