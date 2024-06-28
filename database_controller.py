import os
from datetime import datetime
import re
import cv2
import face_recognition
import psycopg2


class Database:
    def __init__(self, dbname: str = "users", user: str = "postgres", password: str = "",
                 host: str = "127.0.0.1", port: str = "5432"):
        self.connection = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        self.cursor = self.connection.cursor()

    def create_users_table(self) -> None:
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS users_info (
            id SERIAL PRIMARY KEY,
            total_attendance INTEGER,
            last_attendance TIMESTAMP WITH TIME ZONE,
            img_enc NUMERIC[]
        );
        """)
        self.connection.commit()

    def create_records_table(self) -> None:
        self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER,
                    attendance_time TIMESTAMP WITH TIME ZONE
                );
                """)
        self.connection.commit()

    def add_images_to_database(self, directory_path: str) -> None:
        files = [os.path.join(directory_path, file) for file in os.listdir(directory_path)
                 if re.match(r'.*\.(jpg|jpeg|png)', file, flags=re.I)]
        for file in files:
            img = cv2.imread(file)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(img_rgb)
            if enc:
                measurements = enc[0]
                self.cursor.execute("INSERT INTO users_info (total_attendance, last_attendance, img_enc) "
                                    "VALUES (%s, %s, %s);", (0, datetime.now(), measurements.tolist()))
        self.connection.commit()

    def make_sql_request(self, sql_request: str, data: tuple = None) -> None:
        self.cursor.execute(sql_request, data)
        self.connection.commit()

    def fetchall(self) -> list:
        return self.cursor.fetchall()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
        if self.cursor:
            self.cursor.close()
