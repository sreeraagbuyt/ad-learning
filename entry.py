import multiprocessing
import dataUtils as du
import run_ml_model as model
import os
import Constants
import util as u

### Configurations for creating category files ###

main_input_folder = "."
one_hot_encoded_folder = "./encoded"
main_filename = "merged_userProfile_2017-04-30_2017-06-29.csv"
category_folder = "./category_files"
binarized_category_folder = "./binarized_category_files"

### Configurations for running the ml model ###

dataset_folder = "./category_files"
model.report_path = "./report/sub/ml_report"
model.dump_models_folder = "./models/mobilewalla"
model.load_models_folder = "./models/mobilewalla"
model.save_predictions_folder = "./predictions/sub"
model.model_list = ["lr"]
model.threshold_range = map(lambda x: x / 100.0, range(0, 101, 1))
model.test_set_ratio = 0.3


### Fields to be dropped from dataframe initially, other than client_id, impression and click ###

model.drop_fields_initial = []      # Constants.CONTINUOUS_COLUMNS + Constants.CATEGORICAL_VALUES

### Parallelism ###

num_workers_model = 1 
logger = u.getLogger("ProcessLog")



# @params: (dataset_path, load_models, dump_models, vary_threshold, add_random_column)
def run_m(cat):
    logger.info("Running for:" + cat)
    model.run(dataset_folder+"/"+cat+".csv",False,False,True,False)


class ModelRunner:
    def __init__(self):
        #self.categories_to_run = ["AU","FB","RE", "TE", "AP", "CE", "EC", "FA", "HC", "IT", "RT", "SS", "TT", "EN"]
        self.categories_to_run = ["FB|BG","EN|MV","RE|PT","IH|IH","FB|IS","TE|EL","EN|SG","FB|SE","EC|RT","RT|FM","FB|MF","TE|Mt","IT|TM","AU|FW","AU|TW"]

    def runModel(self):
        jobs = []
        pool = multiprocessing.Pool(num_workers_model)
        pool.map(run_m, self.categories_to_run)



if __name__ == '__main__':
    logger.info("In Main")
    multiprocessing.Pool(1).map(du.replaceMissingValues, [[main_input_folder+"/", main_filename]])
    multiprocessing.Pool(1).map(du.oneHotEncode, [[main_input_folder+"/", main_filename.split('.')[0]+"_filled_missing.csv", "./encoded/"]])
    du.mainCategoryExtract(one_hot_encoded_folder+'/',main_filename.split('.')[0]+"_filled_missing.csv",category_folder+'/')
    modelRunner = ModelRunner()
    modelRunner.runModel()
    logger.info("Process Complete")
