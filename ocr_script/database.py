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