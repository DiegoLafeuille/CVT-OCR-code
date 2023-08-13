import sqlite3

class OCRDatabase:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Measurements (
                                measurement_id INTEGER PRIMARY KEY,
                                measurement_name TEXT,
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
    
    def insert_measurement(self, measurement_name, comment):
        self.cursor.execute("INSERT INTO Measurements (measurement_name, comment) VALUES (?, ?)",
                            (measurement_name, comment))
        self.conn.commit()
        return self.cursor.lastrowid
    
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

def setup_database(meas_name, meas_comment, vars):
        
    # Connect to database and create tables if non-existent
    db = OCRDatabase("ocr_script/ocr_database.db")
    db.create_tables()

    # Insert new measurement
    meas_id = db.insert_measurement(meas_name, meas_comment)

    # Insert variables
    variable_ids = []
    for var in vars:
        var_id = db.insert_variable(meas_id, var)
        variable_ids.append({"Variable": var, "ID": var_id})

    return db, meas_id, variable_ids