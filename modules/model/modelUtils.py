from sklearn.externals import joblib
import modelConf as m

class ModelUtils:
    def __init__(self, model_path='/home/buyt-app/ad_learning/models', report_path='/home/buyt-app/ad_learning/reports'):
        self.model_path = model_path
        self.report_path = report_path


    def dump_model(self, model, name):
        joblib.dump(model, self.model_path + "/" + name + ".pkl", compress=True)


    def load_model(self, name):
        return joblib.load(self.model_path + "/" + name + ".pkl")


    def get_ml_model(self, model_code):
        """ Args: model_code For ex: 'lr' """
        return m.ml_models[model_code][0], m.ml_models[model_code][1], m.ml_models[model_code][2]


    def save_feature_importances(self, model, name, model_type):
        f = open(self.report_path+"/"+name+"_importances.csv", 'a+')

        if model_type == "tree":
            importances = model.feature_importances_
            for value in importances:
                f.write(str(value)+'\n')

        if model_type == "reg":
           coef = model.coef_
           for list_ in coef:
                for value in list_:
                    f.write(str(value)+'\n')

        f.write('\n')
        f.close()
