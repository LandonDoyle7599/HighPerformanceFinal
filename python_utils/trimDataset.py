import pandas as pd
import argparse

#read args for input and output file paths
parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input CSV file containing the full dataset.')
parser.add_argument('output_file', type=str, help='Path to the output CSV file for the trimmed dataset.')
args = parser.parse_args()

# read dataset
df = pd.read_csv(args.input_file)

# trim down to 3 columns
columns_to_keep = ['energy', 'speechiness', 'liveness']
df = df[columns_to_keep]

# write trimmed file
df.to_csv(args.output_file, index=False)