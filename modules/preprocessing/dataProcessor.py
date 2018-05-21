import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gc
import warnings
from sklearn.decomposition import IncrementalPCA
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer


class DataProcessor:
    def __init__(self, label, id_="client_id", skipcols=None):
        self.id_ = id_
        self.label = label
        self.skipcols = skipcols if skipcols else []
        warnings.filterwarnings("ignore")


    def drop_dubious_rows(self, df, threshold_missing=0.4):
	""" Accepts a df, drops dubious rows
        
        Args:
            df:  Dataframe

        Returns:
            The updated df

        """
        df.drop_duplicates(subset=[self.id_], keep='first', inplace=True)
        df.dropna(axis=0, thresh=int(df.shape[1]*threshold_missing), inplace=True)
        return df


    def drop_columns_below_thresh(self, df, threshold_missing=0.5):
	""" Accepts a df, drops dubious columns
        
        Args:
            df:  Dataframe

        Returns:
            The updated df

        """
        drop_cols = []
        for col in df:
            if col == self.id_ or col == self.label or col in self.skipcols:
                continue

            ratio_nan = float(df[col].isnull().sum())/float(df[col].shape[0])

            if ratio_nan > threshold_missing:
                drop_cols.append(col)
            
            if len(drop_cols) > 0:
                df.drop(drop_cols, axis=1, inplace=True)

        return df


    def seggregate_columns_by_type(self, df):
	""" Accepts a df, and seggregates each column into one of three types
        
        Args:
            df:  Dataframe

        Returns:
            bool, categorical, continuous columns, The column split

        """

        bool_cols, categorical_cols, continuous_cols = [],[],[]
        for col in df.columns:
            if col == self.id_ or col == self.label or col in self.skipcols:
                continue
            elif df[col].dropna().value_counts().index.isin([0,1]).all():
                    bool_cols.append(col)
            elif df[col].dtype.name == 'object':
                    categorical_cols.append(col)
            else:
                continuous_cols.append(col)
                
        return bool_cols, categorical_cols, continuous_cols


    def handle_missing_values(self, df, threshold_missing=0.4, drop_records=True):
        """ Accepts a df and handles missing values by dropping col/rows containing NA's above a given threshold and replacing na's 
        
        Args:
            df:  Dataframe
            threshold_missing: The threshold to decide if a column/row is to be dropped

        Returns:
            The df without NA's

        """
        threshold_missing = float(threshold_missing)/100. if threshold_missing > 1 else threshold_missing
        if drop_records:
            print("Processing rows...")
            df = self.drop_dubious_rows(df, threshold_missing)
            print(df.shape)
            print("Processing columns...")
            df = self.drop_columns_below_thresh(df, threshold_missing)
            print(df.shape)
        print("Seggregating columns ===> bool, categorical, continuous")
        bool_cols, categorical_cols, continuous_cols = self.seggregate_columns_by_type(df)
        print("Replacing NA's")
        for col in bool_cols:
            df[col].fillna(0, inplace=True)

        for col in categorical_cols:
            mode = df[col].value_counts().idxmax()
            df[col].fillna(mode, inplace=True)

        for col in continuous_cols:
            mean = round(df[col].mean(),2)
            df[col].fillna(mean, inplace=True)

        return df


    def one_hot_encode(self, df):
        """ Accepts a df, finds out categorical columns and encodes them
        
        Args:
            df:  Dataframe

        Returns:
            The one hot encoded df

        """
        
        categorical_cols = []
        print("Finding Categorical columns")

        for col in df.columns:
            if df[col].dtype.name == 'object' and not(col == self.id_ or col == self.label or col in self.skipcols):
                categorical_cols.append(col)
        
        print("Encoding...")
        for col in categorical_cols:
            df[col] = df[col].apply(lambda x: x.strip().lower())
            df = pd.concat([df, pd.get_dummies(df[col])], axis=1)

            try:
                df.drop('none', axis=1, inplace=True)
            except ValueError:
                pass

            try:
                df.drop('na', axis=1, inplace=True)
            except ValueError:
                pass

        df.drop(categorical_cols, axis=1, inplace=True)
        
        if self.label in df:
            df_tmp = df[self.label]
            df.drop(self.label, axis=1, inplace=True)
            df = pd.concat([df,df_tmp], axis=1)

        return df


    def drop_columns(self, df, columns):
        """ Accepts a df and drops the columns specified
        
        Args:
            df:  Dataframe
            columns: Columns to be dropped

        Returns:
            The updated df

        """

        df.drop(columns, axis=1, inplace=True)
        return df


    def min_max_scale_continuous_columns(self, df):
        """ Accepts a df, identifies continuous columns and scales them in the range (0,1)
        
        Args:
            df:  Dataframe

        Returns:
            The scaled df

        """

        scaler = MinMaxScaler()
        bool_cols, categorical_cols, continuous_cols = self.seggregate_columns_by_type(df)
        for col in continuous_cols:
            df[col] = scaler.fit_transform(df[col])
        return df


    def custom_scale_continuous_columns(self, df, start_label, end_label, scaler=100):
        """ Accepts a df, identifies continuous columns and scales it
        
        Args:
            df:  Dataframe
            start_label: The starting label
            end_label: The name of ending index + 1
            scaler: The value by which each column is divided, default:100
        Returns:
            The scaled df

        """
        for col in list(df.loc[:,start_label:end_label].columns.values):
            try:
                df[col].astype(float)
                df[col] = df[col]/scaler
            except:
                pass
        return df


    def balance_data(self, df, labels=[0,1], get_removed=False):
        """ Accepts a df and balances the data according to the class
        
        Args:
            df:  Dataframe
            field: class_label
            labels: The possible values for class label
            get_removed: Return the dropped rows along with the balanced df

        Returns:
            The balanced df

        """
        field = self.label
        df = df.sample(frac=1)

        df_majority = df.loc[(df[field] == labels[0])]

        if type(labels[0]) is int:
            df_minority = df.loc[(df[field] > labels[0])]
        else:
            df_minority = df.loc[(df[field] == labels[1])]

        ratio = float(df_minority[field].count())/float(df_majority[field].count())
        df_majority_downsampled = df_majority.sample(frac=ratio)
        df_majority_removed = None

        if get_removed:
            df_majority_removed = df_majority.loc[~df_majority.index.isin(df_majority_downsampled.index)]

        del df_majority
        del df
        gc.collect()

        df = pd.concat([df_minority, df_majority_downsampled])
        df = df.sample(frac=1)

        if get_removed:
            return df, df_majority_removed

        else:
            return df


    def polarize_ctr_data(self, df, tweak_field, field, labels=[0,1], get_removed=False):
       	""" Accepts a df and selects all positive class rows, selects the negative class rows which are above a given threshold
        
        Args:
            df:  Dataframe
            tweak_field: Field to consider inorder to select negative class greater than mean threshold
            field: class_label
            labels: The possible values for class label
            get_removed: Return the dropped rows along with the balanced df

        Returns:
            The balanced df

        """
        field = self.label
        df = df.sample(frac=1)
        df_minority = df.loc[(df[field] > labels[0])]
        df_majority = df.loc[(df[field] == labels[0])]
        df_majority_tweak = df.loc[(df[tweak_field] > int(df[tweak_field].mean() + 2)) & (df[field] == labels[0])]
        df_majority_removed = None

        if get_removed:
            df_majority_removed = df_majority.loc[~df_majority.index.isin(df_majority_tweak.index)]

        df = pd.concat([df_majority_tweak, df_minority])
        df.sample(frac=1)

        if get_removed:
            return df, df_majority_removed
        else:
            return df


    def reduce_dimensionality(self, path, start_index, end_index, components=5, chunksize=50000):                  
        """ Accepts a df_pointer to chunk and trains PCA incrementally and returns dimensionally reduced df
            
          Args:
                path: path to csv file
                start_index: index of starting label for slicing
                end_index: index of ending label + 1
                components: Percentage by which the components must be reduced
    
            Returns:
                The trained PCA
        """
        df_pointer = pd.read_csv(path, header=0, chunksize=chunksize)
        components = int((end_index - start_index)*components/100)
        ipca = IncrementalPCA(n_components=components, batch_size=10)

        for idx,df in enumerate(df_pointer):
            print("Training Chunk %d"%(idx))                           
            ipca.partial_fit(df.iloc[:,start_index:end_index])

        
        df_pointer = pd.read_csv(path, header=0, chunksize=chunksize)
        df = self.apply_pca(df_pointer, start_index, end_index, ipca, components)
        return df, ipca


    def apply_pca(self, df_pointer, start_index, end_index, ipca, components, chunk=True):            
        """ Accepts a df_pointer and reduces dimensionality of each chunk using pre-trained PCA
             
            Args:
                 df:  Dataframe
                 start_index: index of starting label for slicing
                 end_index: index of ending label + 1
                 ipca: Trained PCA
                 components: Value to which number of features in the specified range is reduced
     
             Returns:
     
        """
        if chunk:
            for idx, df in enumerate(df_pointer):
                print("Transforming Chunk %d"%(idx))
                temp_df = pd.DataFrame(ipca.transform(df.iloc[:,start_index:end_index]), columns=['pca%i' % i for i in range(components)],
                        index=df.index)            
                temp_df = pd.concat([df.iloc[:,:start_index],temp_df, df.iloc[:,end_index:]],axis=1)
                reduced_df = temp_df if idx == 0 else pd.concat([reduced_df, temp_df])
        else:
	    temp_df = pd.DataFrame(ipca.transform(df_pointer.iloc[:,start_index:end_index]), columns=['pca%i' % i for i in range(components)],
                        index=df_pointer.index)
            reduced_df = pd.concat([df_pointer.iloc[:,:start_index],temp_df, df_pointer.iloc[:,end_index:]],axis=1)
            
        return reduced_df

