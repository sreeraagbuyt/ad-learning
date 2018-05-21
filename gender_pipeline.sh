bin/run.sh genderDemographicAnalysis/preprocess.py
mv data/pca/gender_pred_pca_1.csv  data/pca/temp.csv
cat data/pca/temp.csv data/pca/gender_pred_pca_*.csv > data/pca/gender_pred_pca.csv
rm data/pca/temp.csv
bin/run.sh genderDemographicAnalysis/pred_gender.py
cat predictions/prod_gender_logistic_*.csv > predictions/gender_pred.csv
rm data/pca/gender_pred_pca*.csv
rm predictions/prod_gender_logistic_*.csv
