from source.preprocessing import preprocess_data

# Setup
output_dir   = 'preprocessed-data'
ct_info_file = 'ct_info.csv'

# Do preprocessing
preprocess_data(ct_info_file, output_dir, box_size=100, resample=False)