from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

""" Regression Models"""
ml_models = {}
ml_models["etr"] = [ExtraTreesRegressor(n_estimators=50, n_jobs=-1, min_samples_leaf=100), "ETR", False]
ml_models["adb_dtr"] = [AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=30), "ADABoostDTR", False]
ml_models["dtr"] = [DecisionTreeRegressor(), "DTR", False]
ml_models["svr"] = [svm.LinearSVR(), "SVR", False]
ml_models["sgdr"] = [SGDRegressor(), "SDGR", False]
ml_models["rfr"] = [RandomForestRegressor(n_estimators=60, n_jobs=-1), "RFR", False]
ml_models["lr"] = [LogisticRegression( class_weight={0:.5, 1:.5}), "LR", True]
ml_models["gbr"] = [GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=None, random_state=0,
                                              loss='ls'), "GBR", False]

""" Classification Models"""
ml_models["svc"] = [svm.LinearSVC(), "SVC", 0]
ml_models["poly_svc"] = [OneVsRestClassifier(svm.SVC(kernel='poly', coef0=2)), "SVC_poly", 0]
ml_models["rfc"] = [RandomForestClassifier(n_estimators=60, n_jobs=-1), "RFC", True]
ml_models["etc"] = [ExtraTreesClassifier(n_estimators=50, n_jobs=-1, min_samples_leaf=100), "ETC", True]
ml_models["bnb"] = [BernoulliNB(), "BernoulliNB", True]
ml_models["gnb"] = [GaussianNB(), "GaussianNB", True]
ml_models["nc"] = [NearestCentroid(), "Neares_Centroid", False]
ml_models["knn"] = [KNeighborsClassifier(weights='distance', algorithm='kd_tree', n_neighbors=8), "K-Nearest-Neighbour", False]
ml_models["dtc"] = [DecisionTreeClassifier(), "DTC", False]
ml_models["sgdc"] = [SGDClassifier(loss='log', n_iter=200, alpha=.0000001, penalty='l2', learning_rate='invscaling',
                                   power_t=0.5,
                                   eta0=4.0, shuffle=True, n_jobs=-1), "SGD Classifier", True]
ml_models["adb_dtc"] = [AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=50), "ADABoostDTC", True]
ml_models["gbc"] = [GradientBoostingClassifier(n_estimators=310, max_depth=5), "GBC", True]
