import sqlite3

class OCRDatabase:

    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()

############################## Set up functions ##############################

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Measurements (
                                measurement_id INTEGER PRIMARY KEY,
                                measurement_name TEXT UNIQUE,
                                comment TEXT
                            )''')
        
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Variables (
                                variable_id INTEGER PRIMARY KEY,
                                variable_name TEXT,
                                measurement_id INTEGER,
                                FOREIGN KEY (measurement_id) REFERENCES Measurements (measurement_id)
                            )''')
        
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Frames (
                                frame_id INTEGER PRIMARY KEY,
                                measurement_id INTEGER,
                                timestamp TEXT,
                                variable_id INTEGER,
                                value TEXT,
                                FOREIGN KEY (measurement_id) REFERENCES Measurements (measurement_id),
                                FOREIGN KEY (variable_id) REFERENCES Variables (variable_id)
                            )''')
        self.conn.commit()

    def setup_measurement(self, meas_name, meas_comment, vars):
            
        # Insert new measurement
        meas_id = self.insert_measurement(meas_name, meas_comment)

        # Insert variables
        variable_ids = []
        for var in vars:
            var_id = self.insert_variable(meas_id, var)
            variable_ids.append({"Variable": var, "ID": var_id})

        return meas_id, variable_ids

    def insert_measurement(self, measurement_name, comment):
        try:
            self.cursor.execute("INSERT INTO Measurements (measurement_name, comment) VALUES (?, ?)",
                                (measurement_name, comment))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"Measurement with name '{measurement_name}' already exists!")
            return None

    def insert_variable(self, measurement_id, variable_name):
        self.cursor.execute("INSERT INTO Variables (variable_name, measurement_id) VALUES (?, ?)",
                            (variable_name, measurement_id))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_frame_data(self, measurement_id, timestamp, variable_id, value):
        
        self.cursor.execute("INSERT INTO Frames (measurement_id, timestamp, variable_id, value) VALUES (?, ?, ?, ?)",
                            (measurement_id, timestamp, variable_id, value))
        self.conn.commit()

    def close(self):

        self.conn.close()


############################## Look up functions ##############################

    def search_measurement_by_name(self, name):
        self.cursor.execute("SELECT measurement_id, measurement_name FROM Measurements WHERE UPPER(measurement_name) LIKE ?", ('%' + name.upper() + '%',))
        return self.cursor.fetchall()

    def search_measurement_by_id(self, measurement_id):
        self.cursor.execute("SELECT measurement_id, measurement_name FROM Measurements WHERE measurement_id = ?", (measurement_id,))
        return self.cursor.fetchall()

    def search_variable_by_name(self, name):
        self.cursor.execute("SELECT variable_id, variable_name FROM Variables WHERE UPPER(variable_name) LIKE ?", ('%' + name.upper() + '%',))
        return self.cursor.fetchall()

    def search_variable_by_id(self, variable_id):
        self.cursor.execute("SELECT variable_id, variable_name FROM Variables WHERE variable_id = ?", (variable_id,))
        return self.cursor.fetchall()

    def get_measurement_name_by_id(self, measurement_id):
        self.cursor.execute("SELECT measurement_name FROM Measurements WHERE measurement_id = ?", (measurement_id,))
        return self.cursor.fetchone()[0]

    def get_variable_name_by_id(self, variable_id):
        """
        Fetch the variable name associated with a given variable ID.

        :param variable_id: ID of the variable.
        :return: Name of the variable.
        """
        self.cursor.execute("SELECT variable_name FROM Variables WHERE variable_id=?", (variable_id,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        else:
            return None

    def get_variable_names_by_measurement_id(self, measurement_id):
        self.cursor.execute('''SELECT Variables.variable_id, Variables.variable_name
                                  FROM Variables 
                                  WHERE Variables.measurement_id = ? 
                                  ORDER BY Variables.variable_name''', 
                               (measurement_id,))
        return self.cursor.fetchall()

    def get_data_by_measurement_id(self, measurement_id):
        self.cursor.execute('''SELECT Frames.timestamp, Variables.variable_name, Frames.value 
                               FROM Frames 
                               JOIN Variables ON Frames.variable_id = Variables.variable_id 
                               WHERE Frames.measurement_id = ? 
                               ORDER BY Frames.timestamp, Variables.variable_name''', 
                            (measurement_id,))
        return self.cursor.fetchall()

    def get_data_by_variable_id(self, variable_id):
        self.cursor.execute('''SELECT Frames.timestamp, Frames.value 
                               FROM Frames 
                               WHERE Frames.variable_id = ? 
                               ORDER BY Frames.timestamp''', 
                            (variable_id,))
        return self.cursor.fetchall()

    def get_measurement_id_by_variable_id(self, variable_id):
        self.cursor.execute("SELECT measurement_id FROM Variables WHERE variable_id = ?", (variable_id,))
        return self.cursor.fetchone()[0]




def setup_database():
    
    # Connect to database and create tables if non-existent
    db = OCRDatabase("ocr_script/ocr_database.db")
    db.create_tables()

    return db

