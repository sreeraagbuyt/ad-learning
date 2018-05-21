from sklearn.decomposition import IncrementalPCA
from sklearn import decomposition
from sklearn.externals import joblib
import pandas as pd
model_path='/home/buyt-app/ad_learning/models'


def dump_model(model, name):
	joblib.dump(model, model_path + "/" + name + ".pkl", compress=True)

def reduce_dimensionality(path, start_index=30, end_index=1135, chunksize=50000):
        df_pointer = pd.read_csv(path, header=0, chunksize=chunksize, error_bad_lines=False)
        ipca = IncrementalPCA(n_components=400) # Run with large value for n and then decide on n_components using explained_varaince

        for idx,df in enumerate(df_pointer):
            print("Training Chunk %d"%(idx))
            print(df.head())
            ipca.partial_fit(df.iloc[:,start_index:end_index])
	
	print("cumulative sum")
        print(ipca.explained_variance_ratio_.cumsum()) # Select n_components based on explained_variance_cumulative sum
	dump_model(ipca, "PCA_gender_all")



reduce_dimensionality('/home/buyt-app/ad_learning/data/merged_userGenderTrain.csv')
