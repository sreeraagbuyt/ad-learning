from sklearn.model_selection import train_test_split
import reportUtils as mu
from time import time
import xgboost as xgb
import numpy as np
import yaml


class ModelBuilder:
    def __init__(self, train_test_split=0.2):
        conf = yaml.load(open("/home/buyt-app/ad_learning/conf/config.learning.yml"))['PATHS']
        self.split = 0.20 
        self.threshold_range = map(lambda x: x / 100.0, range(0, 101, 1))
        self.report_path = conf['REPORT_PATH']
        self.model_path = conf['MODEL_PATH']
        self.predictions_path = conf['PREDICTIONS_PATH']


    def get_features_labels(self, df, generate=False):
        dataset = df.as_matrix()
        if generate:
            features = dataset[:, :]
            return features
        features = dataset[:, :-1]
        labels = dataset[:, features.shape[1]:].ravel()
        labels = labels.astype(int)
        return features, labels


    def generate_report(self, train_size, test_size, time, name, y_pred, y_test):
        header = "%s Train: %s, Test: %s, time: %s" %(name, str((train_size)), str((test_size)), str(time/60))
        mu.write_report_header(header, self.report_path+"/"+name+".csv")
        mu.vary_threshold(y_pred, y_test, self.report_path+"/"+name+".csv", self.threshold_range)


    def fit_predict(self, df, model, model_name, predicts_prob):
	""" Accepts a df and trains model and generates reports for test data
        
        Args:
            df:  Dataframe
            model: sklearn Model object
            model_name: Name of the model for logging
            predicts_prob: predicts_probability 1 or 0
        
        Returns:
            The trained model

        """
        features, labels = self.get_features_labels(df)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=self.split, random_state=42)
        
        start = time()
        model.fit(X_train[:, 1:], y_train)
        end = time()
        
        y_pred = model.predict_proba(X_test[:, 1:])[:, 1:].ravel() if predicts_prob else model[i].predict(X_test[:, 1:])
        mu.save_metrics(y_test, y_pred, model_name)

        self.generate_report(len(X_train), len(X_test), end-start, model_name, y_pred, y_test)

        return model


    def train_xgb(self, df_pointer, model_name, ml_algo="logistic", scale_pos=8, n_iter=50, early_stop=20, params={}):
	""" Accepts a df and trains model and generates reports for test data
        
        Args:
            df_pointer:  Dataframe text file reader object
            model_name: Name of the model for logging
            ml_algo: linear/logistic
            scale_pos class weight range 1-10
        
        Returns:
    
        """
        test_set, test_labels = None, None
        xgb_model = None
        tot = 0
        for idx,df in enumerate(df_pointer):
            features, labels = self.get_features_labels(df)
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=self.split, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.split, random_state=1)
            dtrain = xgb.DMatrix(X_train[:,1:], label=y_train)
            dtest = xgb.DMatrix(X_test[:,1:])
            deval = xgb.DMatrix(X_val[:,1:], label=y_val)
            test_set = X_test if test_set is None else np.append(test_set, X_test, axis=0)
            test_labels = y_test if test_labels is None else np.append(test_labels, y_test, axis=0)

            if ml_algo == 'linear':
                param = {'max_depth':5, 'eta':0.02, 'silent':1, 'objective':'reg:linear' }
            elif ml_algo == 'logistic':
                if len(params) > 0:
                    param = params
                else:
                    param = {'max_depth':5, 'eta':0.02, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'max_delta_step':4, 'scale_pos_weight': scale_pos}

            watchlist  = [(deval,'eval')]
            num_round = n_iter
            early_stopping_rounds = early_stop
            
            start = time()
            bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds, xgb_model=xgb_model)
            end = time()
            tot += end - start
            
            xgb_model = self.model_path +'/xgb_%s_%s.model'%(ml_algo, idx)
            bst.save_model(xgb_model)
            y_pred = bst.predict(dtest)
            mu.save_metrics(y_test, y_pred, model_name)
        y_pred = bst.predict(xgb.DMatrix(test_set[:,1:]))
        mu.save_metrics(test_labels, y_pred, model_name)
        self.generate_report(len(X_train), len(X_test), tot, model_name, y_pred, test_labels)

        return


    def partial_fit_predict(self, df_pointer, model, model_name, predicts_prob):
 	""" Accepts a df and trains model and generates reports for test data
        
        Args:
            df_pointer:  Dataframe text file reader
            model_name: Name of the model for logging
            predicts_prob: predicts_probability 1 or 0
        
        Returns:
            The trained model

        """
        test_set, test_labels = None, None
        tot = 0
        for df in df_pointer:
            features, labels = self.get_features_labels(df)
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=self.split, random_state=42)
            start = time()
            try:
                model.partial_fit(X_train[:, 1:], y_train, classes=[0,1])
            except TypeError as te:
                model.partial_fit(X_train[:, 1:], y_train)
            end = time()
            tot += end - start
            y_pred = model.predict_proba(X_test[:, 1:])[:, 1:].ravel() if predicts_prob else model.predict(X_test[:, 1:])
            mu.save_metrics(y_test, y_pred, model_name)
            test_set = X_test if test_set is None else np.append(test_set, X_test, axis=0)
            test_labels = y_test if test_labels is None else np.append(test_labels, y_test, axis=0)

        y_pred = model.predict_proba(test_set[:, 1:])[:, 1:].ravel() if predicts_prob else model.predict(test_set[:, 1:])
        mu.save_metrics(test_labels, y_pred, model_name)
        self.generate_report(len(X_train), len(X_test), tot, model_name, y_pred, test_labels)

        return model


    def generate_results_pretrained(self, df, model, model_name, prob=False, gender=False):
 	""" Accepts a df and trained model object and predicts class label
        
        Args:
            df:  Dataframe
            model: Pretrained model
            prob: predict prob if True else predicts class label
        
        Returns:

        """

        features = self.get_features_labels(df, generate=True)
        if prob:
            try:
                y_pred = model.predict_proba(features[:, 1:])[:, 1:].ravel()
            except Exception as e:
                y_pred = model.predict(features[:, 1:])
        else:
            y_pred = model.predict(features[:, 1:])
        
        clids = list(features[:, :1].astype(int).ravel())
        y_pred = list(y_pred)
        if gender:
            y_pred = ['F' if y_p > 0.5 else 'M' for y_p in y_pred]
        mu.save_predictions_against_clid(self.predictions_path+"/"+model_name, clids, y_pred)

