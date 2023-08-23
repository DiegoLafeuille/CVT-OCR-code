
import pandas as pd
import json
import os

# 1. Load the ground truths from the CSV file
ground_truths_df = pd.read_csv('experiment/ground_truths_3.csv', dtype=str)
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
result_files = [f for f in os.listdir("experiment/exp_results") if f.endswith('.json')]
print(len(result_files), " files found")

# 3. Load and process each JSON file
problem_files = []
for file_name in result_files:
    try:
        with open("experiment/exp_results/" + file_name, 'r') as file:
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
    exposure_time = "100" if "exp" not in components[-1] else components[-1].replace("exp", "").replace("k", "")
    
    # 4. Calculate accuracies for each slide and size
    for slide_data in data[1:]:
        image_code = slide_data["Image code"]
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
                'Exposure Time': exposure_time,
                'Size': size,
                'Unprocessed Accuracy': accuracies['Unprocessed'],
                'Processed Accuracy': accuracies['Processed']
            })
        
print("Problem files:")
print(problem_files)

# 5. Store the results in a pandas DataFrame
df = pd.DataFrame(results)
# print(df.head())

# List of all setup parameter combinations
all_lens_distance_combinations = {
    "daheng_6mm": ["35", "50", "75"],
    "daheng_12mm": ["75", "100", "150"],
    "daheng_25mm": ["150", "200", "250"]
}
all_angles = ["0", "45", "60"]
all_brightness = ["0", "50", "100"]

# Image parameters
all_fonts = ["0", "1", "2", "4"]
all_colors = ["0", "1", "2", "3"]
all_sizes = ["Big", "Medium", "Small"]

# Identifying missing combinations
all_combinations = {(lens, distance, brightness, h_angle, v_angle) for lens, distances in all_lens_distance_combinations.items() for brightness in all_brightness for distance in distances for h_angle in all_angles for v_angle in all_angles}
seen_combinations = {(lens, distance, brightness, h_angle, v_angle) for lens, distance, brightness, h_angle, v_angle in zip(df['Lens'], df['Distance to Screen'], df['Screen Brightness'], df['Horizontal Angle'], df['Vertical Angle'])}
missing_combinations = all_combinations - seen_combinations

# print(len(all_combinations))
# print(len(missing_combinations))
# for missing in missing_combinations:
#     print(missing)


new_rows = []
for lens, distance, brightness, h_angle, v_angle in missing_combinations:
    print(f"Introducing 0% for: Lens={lens}, Distance={distance}cm, Brightness={brightness}%, Horizontal Angle={h_angle}°, Vertical Angle={v_angle}°")
    for font in all_fonts:
        for color in all_colors:
            for size in all_sizes:
                new_rows.append({
                    'Font': font,
                    'Color Combination': color,
                    'Lens': lens,
                    'Distance to Screen': distance,
                    'Screen Brightness': brightness,
                    'Horizontal Angle': h_angle,
                    'Vertical Angle': v_angle,
                    'Exposure Time': '100',
                    'Size': size,
                    'Unprocessed Accuracy': 0.0,
                    'Processed Accuracy': 0.0
                })
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

df.to_csv('extended_ocr_accuracy_results.csv', index=False)
