import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()

    input_file = args.input_file

    # histogram based on average

    dataset = pd.read_json(input_file, orient='records', lines=True)

    # Dataset Perspective Score
    dataset_perspective_scores = []
    for row in tqdm(dataset.to_dict(orient='records'), total=len(dataset)):
        mean_score = np.mean(np.asarray(list(row["initial_scores"].values())))
        max_score = np.max(np.asarray(list(row["initial_scores"].values())))
        row["mean_score_initial"] = mean_score
        row["max_score_initial"] = max_score

        row["post_mean_score"] = np.mean(np.asarray(list(row["post_scores"].values())))
        row["post_max_score"] = np.max(np.asarray(list(row["post_scores"].values())))

        # print(row["post_mean_score"]-row["mean_score_initial"])
        print(row["post_max_score"]-row["max_score_initial"])

        dataset_perspective_scores.append(row)

    dataset_perspective_scores = pd.DataFrame(dataset_perspective_scores)

    # # Create histogram
    # plt.figure(figsize=(10, 6))
    # plt.hist(dataset_perspective_scores["max_score_initial"].to_numpy(), bins=10, color='skyblue', edgecolor='black')
    #
    # # Add labels and title
    # plt.xlabel('Perspective Scores (Max)', fontsize=16)
    # plt.ylabel('Frequency', fontsize=16)
    # plt.title('Histogram of Perspective Scores (Max)', fontsize=20)
    #
    # # Display the plot
    # plt.grid(True)
    # plt.savefig('results/histogram_of_dataset.pdf', format='pdf')