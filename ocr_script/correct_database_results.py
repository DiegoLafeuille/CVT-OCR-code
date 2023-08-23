import sqlite3
import csv

def correct_value(value, minimum, maximum):

    # Remove symbols found at the end of strings 
    if value[-1] in ".+-":
        value = value[:-1]

    # Convert to float and check if the original value is already within the range
    try:
        float_value = float(value)
    except:
        if value != "No text recognized":
            print(value)
        return f"Problem: {value}"
    if minimum <= float_value <= maximum:
        return value

    if "." in value:
        return f"Problem: {value}"

    # Check all possible positions for a decimal point to see if one of them results in a value within the range
    valid_corrections = [value[:i] + "." + value[i:] for i in range(1, len(value)) if minimum <= float(value[:i] + "." + value[i:]) <= maximum]

    # If exactly one position works, return the corrected value
    if len(valid_corrections) == 1:
        return valid_corrections[0]
    else:
        return f"Problem: {value}" # If multiple or no positions work, return the original value

def correct_measurement(measurement_id):

    # Get the variable IDs and names for this measurement
    cursor.execute("SELECT variable_id, variable_name FROM Variables WHERE measurement_id = ?", (measurement_id,))
    variables = cursor.fetchall()

    # Get the measurement name for the given measurement_id
    cursor.execute("SELECT measurement_name FROM Measurements WHERE measurement_id = ?", (measurement_id,))
    measurement_name = cursor.fetchone()[0]
    measurement_name = measurement_name.replace(" ", "_")
    output_file = f"M{measurement_id}_{measurement_name}.csv"

    min_max_values = {}
    for variable_id, variable_name in variables:
        minimum = float(input(f'Enter minimum value for {variable_name}: '))
        maximum = float(input(f'Enter maximum value for {variable_name}: '))
        min_max_values[variable_id] = (minimum, maximum)

    # Write the corrected CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Timestamp"] + [var_name for _, var_name in variables])  # Write the headers

        # Query frames for this measurement, sorted by timestamp
        cursor.execute('''SELECT timestamp, variable_id, value FROM Frames WHERE measurement_id = ? ORDER BY timestamp''', (measurement_id,))
        rows = cursor.fetchall()
        current_timestamp = None
        corrected_row = []
        for timestamp, variable_id, value in rows:
            if current_timestamp != timestamp:
                if corrected_row:
                    writer.writerow([current_timestamp] + corrected_row)
                current_timestamp = timestamp
                corrected_row = []
            minimum, maximum = min_max_values[variable_id]
            corrected_row.append(correct_value(value, minimum, maximum))
        if corrected_row:
            writer.writerow([current_timestamp] + corrected_row)

    print(f'File corrected and saved as {output_file}')
    conn.close()

def correct_variable(variable_id):

    # Get the variable name and associated measurement ID
    cursor.execute("SELECT measurement_id, variable_name FROM Variables WHERE variable_id = ?", (variable_id,))
    measurement_id, variable_name = cursor.fetchone()
    
    # Get the measurement name for the given measurement_id
    cursor.execute("SELECT measurement_name FROM Measurements WHERE measurement_id = ?", (measurement_id,))
    measurement_name = cursor.fetchone()[0]
    measurement_name = measurement_name.replace(" ", "_")
    variable_name = variable_name.replace(" ", "_")
    output_file = f"M{measurement_id}_{measurement_name}_V{variable_id}_{variable_name}.csv"

    minimum = float(input(f'Enter minimum value for {variable_name}: '))
    maximum = float(input(f'Enter maximum value for {variable_name}: '))

    # Write the corrected CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Timestamp", variable_name])  # Write the headers

        # Query frames for this variable, sorted by timestamp
        cursor.execute('''SELECT timestamp, value FROM Frames WHERE variable_id = ? ORDER BY timestamp''', (variable_id,))
        for timestamp, value in cursor.fetchall():
            corrected_value = correct_value(value, minimum, maximum)
            writer.writerow([timestamp, corrected_value])

    print(f'File corrected and saved as {output_file}')
    conn.close()



if __name__ == "__main__":
    
    conn = sqlite3.connect("ocr_script/ocr_database.db")
    cursor = conn.cursor()

    while True:
        mode = input("Do you want to correct a measurement or a variable? Enter 'm' or 'v': ")
        
        if mode == 'm':
            measurement_id = int(input("Enter the measurement ID: "))
            correct_measurement(measurement_id)
            break
        elif mode == 'v':
            variable_id = int(input("Enter the variable ID: "))
            correct_variable(variable_id)
            break

        print("Invalid input. Please enter either 'm' or 'v'.")