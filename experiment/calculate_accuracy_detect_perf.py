import pandas as pd
import json

# Load the JSON data
with open("experiment\detect_performance_exp.json", "r") as file:
    data = json.load(file)

# Prepare data for the DataFrame
df_data = {
    "Cuda": [],
    "ROI #": [],
    "Method": [],
    "Average Time (s)": []
}

# Populate the DataFrame data
for entry in data:
    for method, times in entry["Times"].items():
        average_time = sum(times) / len(times)
        df_data["Cuda"].append(entry["Cuda"])
        df_data["ROI #"].append(entry["ROI #"])
        df_data["Method"].append(method)
        df_data["Average Time (s)"].append(average_time)

# Create the DataFrame
df = pd.DataFrame(df_data)

# Save the DataFrame to a CSV file
csv_filepath_pandas = "experiment/average_detection_times.csv"
df.to_csv(csv_filepath_pandas, index=False)

print(df.head())
