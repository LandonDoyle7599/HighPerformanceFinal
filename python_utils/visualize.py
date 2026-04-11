import argparse
import pandas as pd
import matplotlib.pyplot as plt

def visualize_clusters(input_file, output_file):
    df = pd.read_csv(input_file)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.1, 0.1, 0.7, 0.8])

    scatter = ax.scatter(
        df['x'], df['y'], df['z'],
        c=df['c'],
        cmap='viridis'
    )

    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    ax.set_xlabel("Energy", labelpad=10)
    ax.set_ylabel("Speechiness", labelpad=20)
    ax.set_zlabel("Liveness", labelpad=10)

    plt.savefig(
        output_file,
        dpi=300              
    )
    
    
if __name__ == "__main__":
    #read args
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Path to the CSV file containing the clusters.')
    parser.add_argument('output_file', type=str, help='Path to save the output image file (including extension).')
    args = parser.parse_args()
    visualize_clusters(args.input_file, args.output_file)
    