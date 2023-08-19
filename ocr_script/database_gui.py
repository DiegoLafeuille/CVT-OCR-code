import tkinter as tk
from tkinter import ttk, filedialog
import csv
from io import StringIO
import database


class DatabaseApp:

    def __init__(self, root, database):
        self.root = root
        self.db = database
        
        # Title
        self.root.title("Database Navigator")
        
        # Search entry field
        self.search_entry = tk.Entry(self.root, width=40)
        self.search_entry.grid(row=0, column=0, padx=10, pady=10)
        
        # Dropdown for search type
        self.search_type = ttk.Combobox(self.root, values=["Measurement Name", "Measurement ID", "Variable Name", "Variable ID"], state="readonly")
        self.search_type.grid(row=0, column=1, padx=10, pady=10)
        self.search_type.set("Measurement Name")
        
        # Submit button
        self.submit_button = tk.Button(self.root, text="Search", command=self.perform_search)
        self.submit_button.grid(row=0, column=2, padx=10, pady=10)
        
        # Listbox for results
        self.results_listbox = tk.Listbox(self.root, width=60, height=20)
        self.results_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
        
        # Download button
        self.download_button = tk.Button(self.root, text="Download Data", command=self.download_data)
        self.download_button.grid(row=2, column=0, columnspan=3, pady=10)
        
    def perform_search(self):
        # Clear listbox
        self.results_listbox.delete(0, tk.END)
        
        # Get search term and type
        term = self.search_entry.get()
        search_type = self.search_type.get()
        
        # Perform search based on type
        if search_type == "Measurement Name":
            results = self.db.search_measurement_by_name(term)
            for res in results:
                self.results_listbox.insert(tk.END, f"Measurement ID: {res[0]}, Name: {res[1]}")
        elif search_type == "Measurement ID":
            try:
                results = self.db.search_measurement_by_id(int(term))
                for res in results:
                    self.results_listbox.insert(tk.END, f"Measurement ID: {res[0]}, Name: {res[1]}")
            except ValueError:
                self.results_listbox.insert(tk.END, "Invalid Measurement ID!")
        elif search_type == "Variable Name":
            results = self.db.search_variable_by_name(term)
            for res in results:
                self.results_listbox.insert(tk.END, f"Variable ID: {res[0]}, Name: {res[1]}")
        elif search_type == "Variable ID":
            try:
                results = self.db.search_variable_by_id(int(term))
                for res in results:
                    self.results_listbox.insert(tk.END, f"Variable ID: {res[0]}, Name: {res[1]}")
            except ValueError:
                self.results_listbox.insert(tk.END, "Invalid Variable ID!")

    def download_data(self):
        selected = self.results_listbox.curselection()
        if not selected:
            return

        # Extract ID from selected item in listbox
        item = self.results_listbox.get(selected[0])
        if "Measurement ID:" in item:
            meas_id = int(item.split(":")[1].split(",")[0].strip())
            meas_name = self.db.get_measurement_name_by_id(meas_id)
            variables = self.db.get_variable_names_by_measurement_id(meas_id)
            data = self.db.get_data_by_measurement_id(meas_id)
            default_filename = f"M{meas_id}_{meas_name.replace('/', '_').replace(' ', '_').replace('?', '_')}.csv"
            self.save_data_as_csv(data, variables, default_filename)
        elif "Variable ID:" in item:
            var_id = int(item.split(":")[1].split(",")[0].strip())
            var_name = item.split("Name:")[1].strip()
            data = self.db.get_data_by_variable_id(var_id)
            meas_id = meas_id = self.db.get_measurement_id_by_variable_id(var_id)
            meas_name = self.db.get_measurement_name_by_id(meas_id)
            default_filename = f"M{meas_id}_{meas_name.replace('/', '_').replace(' ', '_').replace('?', '_')}_V{var_id}_{var_name.replace('/', '_').replace(' ', '_').replace('?', '_')}.csv"
            self.save_data_as_csv(data, [(var_id, var_name)], default_filename)
            
    def save_data_as_csv(self, data, variables, default_filename):
        # Prepare the data in the desired format
        timestamps = sorted(set([row[0] for row in data]))

        # Check if it's a single variable or multiple variables based on the length of the variables list
        if len(variables) == 1:  # Single variable
            formatted_data = [[timestamp, next((entry[1] for entry in data if entry[0] == timestamp), None)] for timestamp in timestamps]
        else:  # Multiple variables
            formatted_data = []
            for timestamp in timestamps:
                row = [timestamp] + [next((entry[2] for entry in data if entry[0] == timestamp and entry[1] == var[1]), None) for var in variables]
                formatted_data.append(row)

        # Write the data to CSV
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp"] + [f"{var[1]} ({var[0]})" for var in variables])
        writer.writerows(formatted_data)
        
        # Save to file using a file dialog with a default filename
        file_name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], initialfile=default_filename)
        if file_name:
            with open(file_name, "w", newline="") as file:
                file.write(output.getvalue())

        output.close()





if __name__ == "__main__":
    db = database.setup_database()
    root = tk.Tk()
    app = DatabaseApp(root, db)
    root.mainloop()