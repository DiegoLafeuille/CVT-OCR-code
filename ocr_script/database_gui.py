
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import database
import pandas as pd


class DatabaseApp:

    def __init__(self, root, db):
        self.root = root
        self.db = db
        
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
        
        # Listbox for results with multi-selection enabled
        self.results_listbox = tk.Listbox(self.root, width=60, height=20, selectmode=tk.MULTIPLE)
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
        selected_indices = self.results_listbox.curselection()
        
        # Handle if no item is selected
        if not selected_indices:
            return
        
        # Check if all selected measurements have the same variables (by name, not ID)
        variable_sets = []
        for index in selected_indices:
            selected_item = self.results_listbox.get(index)
            if "Measurement ID" in selected_item:
                measurement_id = int(selected_item.split(":")[1].split(",")[0].strip())
                variable_names = [var[1] for var in self.db.get_variable_names_by_measurement_id(measurement_id)]
                variable_sets.append(set(variable_names))
        if variable_sets and len(set(frozenset(s) for s in variable_sets)) > 1:
            messagebox.showerror("Error", "Selected measurements do not have the same variables.")
            return
        
        # Fetch data for selected items and prepare for CSV
        all_data = []
        for index in selected_indices:
            selected_item = self.results_listbox.get(index)
            if "Measurement ID" in selected_item:
                measurement_id = int(selected_item.split(":")[1].split(",")[0].strip())
                data = self.db.get_data_by_measurement_id(measurement_id)
                all_data.extend(data)
            elif "Variable ID" in selected_item:
                variable_id = int(selected_item.split(":")[1].split(",")[0].strip())
                data = self.db.get_data_by_variable_id(variable_id)
                all_data.extend(data)
                
        # Check if the selections are measurements or variables
        if "Measurement ID" in selected_item:
            # Extract unique variable names and timestamps
            variable_names = list(set([data[1] for data in all_data]))
            timestamps = sorted(list(set([data[0] for data in all_data])))

            df = pd.DataFrame(all_data, columns=["Timestamp", "Variable", "Value"])
            df = df.pivot(index='Timestamp', columns='Variable', values='Value')
                
        elif "Variable ID" in selected_item:
            
            # Ensure selected variables have unique name
            variable_names_temp = [self.db.get_variable_name_by_id(int(self.results_listbox.get(index).split(":")[1].split(",")[0].strip())) for index in selected_indices]
            if len(set(variable_names_temp)) != 1:
                messagebox.showerror("Error", "Selected variables must have same name to be coompiled into one file.")
                return
            
            variable_name = variable_names_temp[0]
            timestamps = sorted(list(set([data[0] for data in all_data])))
            timestamps, values = zip(*all_data)
            df = pd.DataFrame(values, index=timestamps, columns=[variable_name])
            df.index.name = "Timestamp"

            
        # Saving the CSV
        filename = ""
        if len(selected_indices) == 1:
            filename = selected_item.split(",")[1].split(":")[1].strip()
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=filename, filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        

        df.sort_values("Timestamp", inplace=True)
        df.to_csv(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    db = database.setup_database()
    app = DatabaseApp(root, db)
    root.mainloop()
