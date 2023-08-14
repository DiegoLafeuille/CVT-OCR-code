import sqlite3
import csv

def correct_value(value, minimum, maximum):
    # Convert to float and check if the original value is already within the range
    try:
        float_value = float(value)
    except:
        return value
    if minimum <= float_value <= maximum:
        return float_value

    # Check all possible positions for a decimal point to see if one of them results in a value within the range
    valid_corrections = [float(value[:i] + "." + value[i:]) for i in range(1, len(value)) if minimum <= float(value[:i] + "." + value[i:]) <= maximum]

    # If exactly one position works, return the corrected value
    if len(valid_corrections) == 1:
        return valid_corrections[0]
    else:
        return float_value # If multiple or no positions work, return the original value

def correct_measurement(measurement_id, output_file):

    # Get the variable IDs and names for this measurement
    cursor.execute("SELECT variable_id, variable_name FROM Variables WHERE measurement_id = ?", (measurement_id,))
    variables = cursor.fetchall()

    min_max_values = {}
    for variable_id, variable_name in variables:
        minimum = float(input(f'Enter minimum value for {variable_name}: '))
        maximum = float(input(f'Enter maximum value for {variable_name}: '))
        min_max_values[variable_id] = (minimum, maximum)

    # Write the corrected CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Timestamp"] + [var_name for _, var_name in variables]) # Write the headers

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

def correct_variable(variable_id, output_file):

    # Get the variable name
    cursor.execute("SELECT variable_name FROM Variables WHERE variable_id = ?", (variable_id,))
    variable_name = cursor.fetchone()[0]
    minimum = float(input(f'Enter minimum value for {variable_name}: '))
    maximum = float(input(f'Enter maximum value for {variable_name}: '))

    # Write the corrected CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Timestamp", variable_name]) # Write the headers

        # Query frames for this variable, sorted by timestamp
        cursor.execute('''SELECT timestamp, value FROM Frames WHERE variable_id = ? ORDER BY timestamp''', (variable_id,))
        for timestamp, value in cursor.fetchall():
            corrected_value = correct_value(value, minimum, maximum)
            writer.writerow([timestamp, corrected_value])

    print(f'File corrected and saved as {output_file}')
    conn.close()

if __name__ == "__main__":
    
    mode = input("Do you want to correct a measurement or a variable? Enter 'measurement' or 'variable': ")
    output_file = input("Enter the output CSV file name: ")
    
    conn = sqlite3.connect("ocr_script/ocr_database.db")
    cursor = conn.cursor()
    
    if mode == 'measurement':
        measurement_id = int(input("Enter the measurement ID: "))
        correct_measurement(measurement_id, output_file)
    elif mode == 'variable':
        variable_id = int(input("Enter the variable ID: "))
        correct_variable(variable_id, output_file)
    else:
        print("Invalid input. Please enter either 'measurement' or 'variable'.")