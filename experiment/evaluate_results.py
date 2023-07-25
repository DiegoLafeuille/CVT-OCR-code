import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import json
import os

def import_experiment_results(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    parameters = data[0]
    results = data[1:]

    df_results = pd.DataFrame(results)
    return parameters, df_results

def barplot_results_original(result_dfs):

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

def barplot_results(result_dfs):
    labels = ['Big', 'Medium', 'Small']
    num_rows = 4
    num_columns = 4
    color_map = {result_dfs[0]['File']: 'blue', result_dfs[1]['File']: 'orange'}  # map file names to colors

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 20), gridspec_kw={'hspace': 0.4})

    # Flatten the array of Axes for easier iteration
    axs = axs.flatten()

    # Loop through all image codes
    for i, image_code in enumerate(result_dfs[0]["Averages"]["Image code"].unique()):
        ax = axs[i]
        x = np.array([1,2,3])  # labels positions
        width = 0.15  # the width of the bars

        # Calculate positions for each set of bars
        bar_pos = [x - 3*width/2, x + width/2]

        for j, result in enumerate(result_dfs):
            averages = result["Averages"].loc[result["Averages"]["Image code"] == image_code].to_dict(orient="records")[0]

            unproc_accs = [averages[acc_type] for acc_type in ["Big unproc acc", "Medium unproc acc", "Small unproc acc"]]
            proc_accs = [averages[acc_type] for acc_type in ["Big proc acc", "Medium proc acc", "Small proc acc"]]

            pos_unproc = bar_pos[0] + j*width
            pos_proc = bar_pos[1] + j*width
            ax.bar(pos_unproc, unproc_accs, width, color=color_map[result['File']])
            ax.bar(pos_proc, proc_accs, width, color=color_map[result['File']])

        ax.set_ylabel('Accuracy', fontsize=8)
        ax.set_title(f'Image {image_code}', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)

    # fig.tight_layout()
    plt.show()




def main():

    ground_truths_df = pd.read_csv("experiment/ground_truths_3.csv", dtype=str)
    # print(ground_truths_df)

    # files = list(os.walk("experiment/exp_results"))[0][2]
    files = ["daheng_12mm_100_0_45_0.json", "daheng_12mm_100_0_45_0_exp125k.json"]
    # files = ["daheng_12mm_100_0_0_0.json"]
    result_dfs = []

    for file in files:
        
        filepath = "experiment/exp_results/" + file
        parameters, df_results = import_experiment_results(filepath)

        print(f"File name: {file}")
        # # Print the experiment external parameters
        # print("Experiment Parameters:")
        # for key, value in parameters.items():
        #     print(f"{key}: {value}")
        # print()

        for index, slide in df_results.iterrows():
            slide_truths = ground_truths_df.loc[ground_truths_df["Code"] == slide["Image code"]].to_dict(orient='records')[0]
            # print(slide_truths)            

            # print(f"Ground truth for image {slide['Image code']}: {slide_truths}")

            big_unprocessed_acc = slide["Big"]["Unprocessed"].count(slide_truths["Big"])
            big_processed_acc = slide["Big"]["Processed"].count(slide_truths["Big"])
            df_results.at[index, "Big unproc acc"] = big_unprocessed_acc
            df_results.at[index, "Big proc acc"] = big_processed_acc

            medium_unprocessed_acc = slide["Medium"]["Unprocessed"].count(slide_truths["Medium"])
            medium_processed_acc = slide["Medium"]["Processed"].count(slide_truths["Medium"])
            df_results.at[index, "Medium unproc acc"] = medium_unprocessed_acc
            df_results.at[index, "Medium proc acc"] = medium_processed_acc

            small_unprocessed_acc = slide["Small"]["Unprocessed"].count(slide_truths["Small"])
            small_processed_acc = slide["Small"]["Processed"].count(slide_truths["Small"])
            df_results.at[index, "Small unproc acc"] = small_unprocessed_acc
            df_results.at[index, "Small proc acc"] = small_processed_acc

            # print(f"Unprocessed accuracy = {unprocessed_acc} / {len(slide['Unproc img results'])}")
            # print(f"Processed accuracy = {processed_acc} / {len(slide['Proc img results'])}")

        average_df = df_results[[
            "Image code", 
            "Big unproc acc", "Big proc acc", 
            "Medium unproc acc", "Medium proc acc", 
            "Small unproc acc", "Small proc acc"
        ]]
        average_dict = {"File": file, "Averages": average_df}
        result_dfs.append(average_dict)
    

    # Plot results in result_dfs as bar plots
    barplot_results(result_dfs)




if __name__ == '__main__':
    main()