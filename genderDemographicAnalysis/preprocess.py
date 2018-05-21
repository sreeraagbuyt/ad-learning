from modules.preprocessing import dataProcessor
import pandas as pd
from modules.model import modelBuilder, modelUtils
import yaml
import multiprocessing
import os


def preprocess_without_memory_leak(df,idx,pca):
    start_index_apps = 28     # index of apps data after dropping/processing categorical fields
    end_index_apps = 1133     # end index of apps data after dropping/processing catergorical fields + 1
    n_components = 250        # only used for getting header names now (pca_1, pca_2, pca_3)

    # 1) Fill na with same mean/mode values used while training
    cols = conf.keys()
    dp = dataProcessor.DataProcessor(label)
    for col in cols:
        df[col].fillna(conf[col], inplace=True)
    
    # 2) Transform label to numeric field
    if label in df.columns:
        df[label] = df[label].apply(lambda x: 0 if x == 'M' else 1)

    # 3) One hot encode the categorical columns if any are present
    tdf = dp.one_hot_encode(df)
    
    # 4) Transform the app based features using trained pca feature selector
    print("Transforming Chunk "+str(idx))
    tdf = dp.apply_pca(tdf, start_index_apps, end_index_apps, pca, n_components, chunk=False)
    
    # Write to file
    set_header = True if idx == 1 else False
    tdf.to_csv(BASE_PATH+"/data/pca"+file_to_csv+str(idx)+".csv", header=set_header, index=False)


BASE_PATH = "/home/buyt-app/ad_learning"
conf = yaml.load(open(BASE_PATH+'/data/config.na.yml'))
file_to_csv = "/gender_pred_pca_"
final_file = "/gender_pred_pca"
label = "gender"
pca_dir = "data/pca"
filename = "/data/merged_userGenderPredict.csv.gz"
modelname = "PCA_gender_optimal"
chunksize = 100000
num_workers = 1
idx = 0

mu = modelUtils.ModelUtils()
pca = mu.load_model(modelname)
df_pointer = pd.read_csv(BASE_PATH+filename, chunksize=chunksize, header=0, error_bad_lines=False,compression='gzip')

for df in df_pointer:
    idx += 1
    new_columns = df.columns.values; new_columns[1] = 'NAG'; df.columns = new_columns
    pool = multiprocessing.Pool(num_workers, preprocess_without_memory_leak, (df,idx,pca,))
    pool.close()
    pool.join()

"""
pca_files = sorted(os.listdir(pca_dir))
with open(pca_dir + final_file + ".csv", "wb") as outfile:
    for filename in pca_files:
        with open(filename, "rb") as infile:
            outfile.write(infile.read())
"""
