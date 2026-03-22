import pandas as pd

# read dataset
df = pd.read_csv('tracks_features.csv')

# trim down to 3 columns
columns_to_keep = ['energy', 'speechiness', 'liveness']
df = df[columns_to_keep]

# write trimmed file
df.to_csv('trimmed_tracks_features.csv', index=False)