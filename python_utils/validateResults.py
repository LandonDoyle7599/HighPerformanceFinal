import pandas as pd
import argparse

def compareFiles(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1.equals(df2)

if __name__ == "__main__":
    #read args
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file1', type=str, help='Path to the first CSV file containing the clusters.')
    parser.add_argument('input_file2', type=str, help='Path to the second CSV file containing the clusters.')
    args = parser.parse_args()

    df1 = pd.read_csv(args.input_file1)
    df2 = pd.read_csv(args.input_file2)
    if compareFiles(args.input_file1, args.input_file2):
        print("The files are identical.")
    else:
        print("The files are different.")