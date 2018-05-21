import pandas as pd
import yaml

# Use this to find mean/mode of columns in training set and save to conf
# The values in conf may later be used for imputing na values in test set

BASE_PATH = "/home/buyt-app/ad_learning"
cols_to_impute = "NAG,w,4G,3G,2G,english,hindi,tamil,telugu,malayalam,kannada,entertainment,business,education,regional,technology,sports,Media & Video,News & Magazines,Travel & Local,Social,Music & Audio,Photography,Entertainment,Shopping,Books & Reference,Education,ratio_imgs_clicked"
filename = "/data/merged_userGenderTrain.csv"
conf_na = {}

df = pd.read_csv(BASE_PATH+filename, header=0, usecols=cols_to_impute.split(','), error_bad_lines=False)
for col in df.columns:
    conf_na[col] = round(df[col].mean(),4)

with open(BASE_PATH+'/conf/config.na.yml', 'w') as outfile:
        yaml.dump(conf_na, outfile, default_flow_style=False)

