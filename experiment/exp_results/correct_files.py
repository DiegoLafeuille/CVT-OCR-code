import os
import json

def swap_angle_values():
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            print("Correcting ", filename)
            # Load the data from the json file
            with open(filename, 'r') as json_file:
                data = json.load(json_file)

            # Ensure there is at least one dictionary in the list
            if len(data) > 0:
                # Swap the 'Horizontal angle' and 'Vertical angle' values
                data[0]['Horizontal angle'], data[0]['Vertical angle'] = data[0]['Vertical angle'], data[0]['Horizontal angle']

            # Write the modified data back to the json file
            with open(filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)

# Call the function
swap_angle_values()
