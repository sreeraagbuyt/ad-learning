CONTINUOUS_COLUMNS = ["NA", "w", "4G", "3G", "2G", "english", "hindi", "tamil", "telugu", "malayalam", "kannada", "entertainment", "business", "education", "politics", "general", "cricket","travel", "regional", "technology", "sports", "astrology", "health & lifestyle", "automobile", "festival", "Media & Video", "News & Magazines", "Travel & Local", "Social", "Music & Audio", "Photography", "Entertainment", "Shopping", "Books & Reference", "Education"]

CATEGORICAL_COLUMNS = ["location", "master_source", "device"]

END_INDEX_CATEGORY = -3

CATEGORICAL_VALUES = ["Tier1","Tier2","Tier3","Tier4","APK","Adwords","Books","Emp Referral","Facebook","GP","GP-Incent","GP-Non Incent","Organic","Other_Organic","Preburn","Referral","Taboola","WAP","inorganic","organic","preburn","HIGH","LOW","MID"]

fields_to_check = "in_re_cons,in_gd_fema,in_gd_mima,in_re_hots,in_re_yohi,in_gd_male,in_gd_lowi,in_en_musi,in_tr_busi,in_en_soci,in_gd_avin,in_gd_mife,in_gd_1924,in_gd_hiin,in_tr_leis,in_au_aubu,in_gd_3549,in_gd_2534,in_re_inte,in_gd_olde,in_ls_epmo,in_gd_1318"

FEATURES = CONTINUOUS_COLUMNS +  fields_to_check.split(',') + CATEGORICAL_VALUES
START_INDEX_CATEGORY = len(CONTINUOUS_COLUMNS) + len(CATEGORICAL_VALUES) + 1 #+ len(fields_to_check.split(',')) #The starting index of category_i (EN|SM_i)

if __name__ == "__main__":
    print START_INDEX_CATEGORY
