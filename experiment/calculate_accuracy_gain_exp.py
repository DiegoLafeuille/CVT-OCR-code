
import pandas as pd
import json
import os

# 1. Load the ground truths from the CSV file
ground_truths_df = pd.read_csv('experiment/ground_truths/ground_truths_3.csv', dtype=str)
ground_truths = ground_truths_df.set_index('Code').to_dict(orient='index')

# 2. Define a function to calculate word accuracy
def calculate_word_accuracy(ground_truth, ocr_data):
    accuracies = {}
    for key in ['Unprocessed', 'Processed']:
        correct_count = sum([1 for result in ocr_data[key] if result == ground_truth])
        accuracies[key] = correct_count / len(ocr_data[key])
    return accuracies

results = []

# List of JSON files (you can modify this list as needed)
result_files = [f for f in os.listdir("experiment/exp_results/gain_exp") if f.endswith('.json')]
print(len(result_files), " files found")

# 3. Load and process each JSON file
problem_files = []
for file_name in result_files:
    try:
        with open("experiment/exp_results/gain_exp/" + file_name, 'r') as file:
            data = json.load(file)
    except:
        problem_files.append(file_name)
        continue
    
    # Extract parameters from file name based on "mm" suffix for the lens
    lens_end_idx = file_name.find("mm") + 2
    lens = file_name[:lens_end_idx]
    components = file_name[lens_end_idx + 1:].replace('.json', '').split('_')
    distance = components[0]
    brightness = components[1]
    horizontal_angle = components[2]
    vertical_angle = components[3]
    
    # 4. Calculate accuracies for each slide and size
    for slide_data in data[1:]:
        image_code = slide_data["Image code"]
        exposure = slide_data["Exposure"]
        gain = slide_data["Gain"]
        for size in ['Big', 'Medium', 'Small']:
            accuracies = calculate_word_accuracy(ground_truths[image_code][size], slide_data[size])
            results.append({
                'Font': image_code[0],
                'Color Combination': image_code[1],
                'Lens': lens,
                'Distance to Screen': distance,
                'Screen Brightness': brightness,
                'Horizontal Angle': horizontal_angle,
                'Vertical Angle': vertical_angle,
                'Exposure Time': exposure,
                'Gain': gain,
                'Size': size,
                'Unprocessed Accuracy': accuracies['Unprocessed'],
                'Processed Accuracy': accuracies['Processed']
            })
        
print("Problem files:")
print(problem_files)

# 5. Store the results in a pandas DataFrame
df = pd.DataFrame(results)
print(df.head())

df.to_csv('experiment/exp_results/exp_gain_results.csv', index=False)
