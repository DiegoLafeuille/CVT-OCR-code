import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import json

def import_experiment_results(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    parameters = data[0]
    results = data[1:]

    df_results = pd.DataFrame(results)
    return parameters, df_results

def barplot_results(result_dfs):

    fig, axs = plt.subplots(nrows=2, sharex=True)
    x = np.arange(len(result_dfs[0]["Averages"]))

    x_tick_labels = result_dfs[0]["Averages"]["Image code"]  # Use the first file for x-axis tick labels

    bar_width = 0.35  # Width of the bars
    offset = bar_width * len(result_dfs) / 2  # Offset to center the bars

    for idx, result in enumerate(result_dfs):
        file = result["File"]
        averages = result["Averages"]

        axs[0].bar(x - offset + idx * bar_width, averages["Unproc acc"], bar_width, label=f"{file}")
        axs[1].bar(x - offset + idx * bar_width, averages["Proc acc"], bar_width, label=f"{file}")

    axs[0].set_ylabel("Unproc acc")
    axs[1].set_ylabel("Proc acc")
    axs[-1].set_xlabel("Image code")

    # Set x-axis tick labels as categorical data
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(x_tick_labels, rotation='vertical')

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main():

    ground_truths_df = pd.read_csv("experiment/ground_truths_new.csv", dtype={"Code": str, "Ground truth": str})
    # print(ground_truths_df)


    files = ["daheng_6mm_50_3_0_0", "daheng_12mm_100_3_0_0"]
    result_dfs = []

    for file in files:
        
        filepath = "experiment/exp_results/" + file + ".json"
        parameters, df_results = import_experiment_results(filepath)

        print(f"File name: {file}")
        # # Print the experiment external parameters
        # print("Experiment Parameters:")
        # for key, value in parameters.items():
        #     print(f"{key}: {value}")
        # print()

        for index, slide in df_results.iterrows():
            ground_truth = ground_truths_df.loc[ground_truths_df["Code"] == slide["Image code"]]["Ground truth"].item()
            # print(f"Ground truth for image {slide['Image code']}: {ground_truth}")

            unprocessed_acc = slide["Unproc img results"].count(ground_truth)
            df_results.at[index, "Unproc acc"] = unprocessed_acc
            processed_acc = slide["Proc img results"].count(ground_truth)
            df_results.at[index, "Proc acc"] = processed_acc

            # print(f"Unprocessed accuracy = {unprocessed_acc} / {len(slide['Unproc img results'])}")
            # print(f"Processed accuracy = {processed_acc} / {len(slide['Proc img results'])}")

        average_df = df_results[["Image code", "Unproc acc", "Proc acc"]]
        average_dict = {"File": file, "Averages": average_df}
        result_dfs.append(average_dict)

    # Plot results in result_dfs as bar plots
    barplot_results(result_dfs)

    



if __name__ == '__main__':
    main()