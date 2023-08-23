import csv
import argparse

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

    # Check if string already contains a dot
    if "." in value:
        return f"Problem: {value}"

    # Check all possible positions for a decimal point to see if one of them results in a value within the range
    valid_corrections = [value[:i] + "." + value[i:] for i in range(1, len(value)) if minimum <= float(value[:i] + "." + value[i:]) <= maximum]

    # If exactly one position works, return the corrected value
    if len(valid_corrections) == 1:
        return valid_corrections[0]
    else:
        return f"Problem: {value}"  # If multiple or no positions work, return the original value

def correct_csv(input_file):
    # Read the input CSV
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        rows = list(reader)

    # Get the min-max values for each variable
    min_max_values = {}
    for header in headers[1:]:
        minimum = float(input(f'Enter minimum value for {header}: '))
        maximum = float(input(f'Enter maximum value for {header}: '))
        min_max_values[header] = (minimum, maximum)

    # Correct the values in the rows
    corrected_rows = []
    for row in rows:
        timestamp = row[0]
        corrected_values = [correct_value(value, *min_max_values[header]) for value, header in zip(row[1:], headers[1:])]
        corrected_rows.append([timestamp] + corrected_values)

    # Write the corrected CSV
    output_file = input_file.split('.')[0] + "_corrected.csv"
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)  # Write the headers
        writer.writerows(corrected_rows)

    print(f'File corrected and saved as {output_file}')

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Correct values in a CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file to be corrected.')
    args = parser.parse_args()

    correct_csv(args.csv_file)
