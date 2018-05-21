from modules.model import modelBuilder, modelUtils
import pandas as pd
from modules.preprocessing import dataProcessor


path = "/home/buyt-app/ad_learning/data/pca"
filename = "/merged_train_gender.csv"
chunksize = 200000
label = "gender"

df = pd.read_csv(path+filename, header=0, chunksize=chunksize, error_bad_lines=False)
dp = dataProcessor.DataProcessor(label)
mb = modelBuilder.ModelBuilder()

#   n_iter = num_rounds = number of trees, use early stop to prevent overfitting
#   scale_pos = sum(negative_class)/sum(positive_class) to handle imbalance

#   prints validation log-loss at each round of training to sys.out also \
#   logs the metrics in the logs folder & finally predicts and generates  \
#   Precision Recall values at different thresholds and saves the final model in models folder 

#   param = {'max_depth':5, 'eta':0.02, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', \
#            'max_delta_step':4, 'scale_pos_weight': scale_pos}

mb.split = 0.20  # Train Test Validation Split. (Val set used to optimize loss function, Test set records result after each chunk and at the end)
mb.train_xgb(df, "gender_xgb", ml_algo="logistic", scale_pos=4, n_iter=1000, early_stop=60)

