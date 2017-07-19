import pandas as pd
from pandas import Series
import Constants
import os
import util as u

logger = u.getLogger("ProcessLog")

def replaceMissingValues(args):
    """
        Replace missing values with the mean for continuous columns, mode for categorical columns
    """
    #TODO if the number of missing values is greter than a threshold drop the column
    main_input_folder = args[0]
    file_name = args[1]
    dtype_dict = {}
    
    for c in Constants.CONTINUOUS_COLUMNS:
        dtype_dict[c] = float
    logger.info("In replaceMissingValues(): Reading:"+file_name)
    df = pd.read_csv(main_input_folder + file_name, header=0, dtype = dtype_dict)
    logger.info("Read Complete")
    
    seg_fields = Constants.fields_to_check.split(',')
    
    try:
        df.dropna(subset=seg_fields,inplace=True)
    except ValueError:
        pass

    df.drop_duplicates(subset=['client_id'], keep='first', inplace=True)
    
    for i in Constants.CATEGORICAL_COLUMNS:
        mode = df[i].value_counts().idxmax()
        df[i].fillna(mode, inplace=True)
    
    for i in Constants.CONTINUOUS_COLUMNS:
        mean = round(df[i].mean(),2)
        df[i].fillna(mean, inplace=True)
    
    df.to_csv(main_input_folder + file_name.split('.')[0] + '_filled_missing.csv', index=False, mode='w+')
    

def oneHotEncode(args): 
    """
        One Hot Encode the categorical columns [device,master_source,location] for each ad-category file
    """
    category_folder = args[0]
    file_name = args[1]
    write_path = args[2]

    logger.info("In oneHotEncode():"+file_name)
    df_entire = pd.read_csv(category_folder + file_name, header=0, chunksize=8000000)
    cnt = 0

    for df in df_entire:
        cnt += 1
        logger.info("Chunk "+str(cnt)+'\n')

        df_rest = df.ix[:,'EN|SM_i':]                           # slice the df from cat_impression_click
        df = df.ix[:,'client_id':'device']                      # userData df
        for i in Constants.CATEGORICAL_COLUMNS:
            df = pd.concat([df, pd.get_dummies(df[i])], axis=1) # get Dummies and concat oneHotencoded values to df

            try:
                df.drop('none', axis=1, inplace=True)
            except ValueError:
                pass

            df.drop(i, axis=1, inplace=True)                    # drop the original categorical column
        
        if cnt == 1:
            header = list(df.columns.values)

        current_chunk_header = list(df.columns.values)

        for field in current_chunk_header:
            if field not in header:
                df.drop(field, axis=1, inplace=True)            # drop the new fields which may appear in the new chunk

        for field in header:
            if field not in current_chunk_header:
                df[field] = 0                                   # add field to the current chunk if it isn't present
                
        df = df[header]                                         # re-arrange the columns in the df

        df = pd.concat([df, df_rest], axis=1)

        if cnt == 1:
            df.to_csv(write_path + file_name, index=False, mode='a+', header=True)

        else:
            df.to_csv(write_path + file_name, index=False, mode='a', header=False)


def mainCategoryExtract(main_input_folder, main_input_file, write_path):
    """
        Extract and save the category files from the merged User Profile
    """
    logger.info("In mainCategoryExtract():")
    f = open(main_input_folder + main_input_file)
    header = f.readline().split(',')
    f.close()
    cat_impression_click = header[Constants.START_INDEX_CATEGORY:Constants.END_INDEX_CATEGORY]
    fields = ','.join(header[:Constants.START_INDEX_CATEGORY]) + ',' + 'impression' + ',' + 'click' # fields to be written
    print fields
    filenames = set()

    for cat in cat_impression_click:
        filenames.add(cat.split('_')[0])                             # set of filenames (main categories '|' sub_categories '_')

    for file_ in filenames:                                          # write header to each file
        f = open(write_path + file_ + '.csv', 'w+')
        f.write(fields)
        f.write('\n')
        f.close()

    with open(main_input_folder + main_input_file, 'r') as f:
        next(f)
        for line in f:
            data = line.split(',')
            user_data = ','.join(data[:Constants.START_INDEX_CATEGORY])
            imp_click_dict = {}
            files_to_write = set()
            for i in range(Constants.START_INDEX_CATEGORY, len(header) - 4, 2):
                #key = header[i].split('|')[0]                        # key is the mainCategory name
                key = header[i].split('_')[0]
                if int(data[i]) > 0:                                 # check if impression > 0
                    files_to_write.add(key)
                    if key not in imp_click_dict:
                        imp_click_dict[key] = {}
                        imp_click_dict[key]['i'] = int(data[i])      # initialize impression
                        imp_click_dict[key]['c'] = int(data[i + 1])  # initialize click
                    else:
                        imp_click_dict[key]['i'] += int(data[i])     # update impression
                        imp_click_dict[key]['c'] += int(data[i + 1]) # update click

            for file_ in files_to_write:
                if file_ in ["AN", "SD"]:
                    continue

                fp = open(write_path + file_ + '.csv', 'a+')
                write_csv_data = user_data + ',' + str(imp_click_dict[file_]['i']) + ',' + str(
                    imp_click_dict[file_]['c'])
                fp.write(write_csv_data + '\n')
                fp.close()
    logger.info("mainCategoryExtract():Complete")


if __name__ == '__main__':
    replaceMissingValues('./','merged_userProfile_2017-04-30_2017-06-29.csv')
    #oneHotEncode('./','merged_userProfile_2017-04-30_2017-06-29_filled_missing.csv','./encoded/')
    #mainCategoryExtract('./encoded/','merged_userProfile_2017-04-30_2017-06-29_filled_missing.csv','./encoded/cat_files/')
