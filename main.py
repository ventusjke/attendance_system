import json
import sys
import cv2
import face_recognition
import numpy as np
from datetime import datetime
from database_controller import Database


def compare_faces(faces_list: list[np.ndarray], face: np.ndarray, tolerance: float = 0.6) -> (bool, int):
    face_distances = face_recognition.face_distance(faces_list, face)
    match_index = np.argmin(face_distances) if face_distances.size else -1
    return (match_index >= 0 and face_distances[match_index] <= tolerance), match_index


def main(database_settings: dict, recognize_counter_limit: int, remember_counter_limit: int, compression_ratio: int,
         frame_counter_limit: int, camera_channel: int, exit_key: int,
         text_color: tuple = (0, 255, 0), frame_color: tuple = (255, 0, 0)) -> None:
    database = Database(**database_settings)
    database.create_users_table()
    database.create_records_table()
    # database.add_images_to_database("images/")
    camera = cv2.VideoCapture(camera_channel)

    frame_counter = 0
    known_faces_counter = {}
    unknown_faces_counter = []
    face_locations = []
    face_names = []
    while True:
        res, frame = camera.read()
        flipped_frame = cv2.flip(frame, 1)
        frame_counter += 1

        if frame_counter >= frame_counter_limit:
            database.make_sql_request("SELECT id, img_enc FROM users_info")
            faces = dict(database.fetchall())
            known_faces = np.array(list(faces.values())).astype(float)
            known_faces_name = list(faces.keys())

            small_flipped_frame = cv2.resize(flipped_frame, (0, 0), fx=1/compression_ratio, fy=1/compression_ratio)
            rgb_flipped_frame = cv2.cvtColor(small_flipped_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_flipped_frame)
            face_encodings = face_recognition.face_encodings(rgb_flipped_frame, face_locations)

            face_names = []
            temp_known_counter = {}
            temp_unknown_counter = []
            for face_enc in face_encodings:
                res, index = compare_faces(known_faces, face_enc)
                name = "Unknown"
                if res:
                    name = known_faces_name[index]
                    temp_known_counter[name] = known_faces_counter.get(name, 0) + 1
                else:
                    unknown_faces = [face for count, face in unknown_faces_counter]
                    res, index = compare_faces(unknown_faces, face_enc)
                    count = unknown_faces_counter[index][0] if res else 0
                    temp_unknown_counter.append((count + 1, face_enc))
                face_names.append(str(name))

            known_faces_counter = temp_known_counter
            names = []
            for name in known_faces_counter:
                if known_faces_counter[name] >= remember_counter_limit:
                    names.append(name)
                    database.make_sql_request("UPDATE users_info "
                                              "SET total_attendance = total_attendance + 1, last_attendance = %s "
                                              "WHERE id = %s;",
                                              (datetime.now(), name))
                    database.make_sql_request("INSERT INTO records (id, attendance_time) "
                                              "VALUES (%s, %s);", (name, datetime.now()))
                    print(f"{name} attendance was approved!")
            for name in names:
                known_faces_counter.pop(name)

            unknown_faces_counter = temp_unknown_counter
            indexes = []
            for i in range(len(unknown_faces_counter)):
                count, unknown_face = unknown_faces_counter[i]
                if count >= recognize_counter_limit:
                    indexes.append(i)
                    database.make_sql_request("INSERT INTO users_info (total_attendance, last_attendance, img_enc) "
                                              "VALUES (%s, %s, %s);", (1, datetime.now(), unknown_face.tolist()))
                    database.make_sql_request("SELECT MAX(id) FROM users_info;")
                    name = database.fetchall()[0]
                    database.make_sql_request("INSERT INTO records (id, attendance_time) "
                                              "VALUES (%s, %s);", (name, datetime.now()))
                    print(f"new face added to db!")
            for index in indexes:
                unknown_faces_counter.pop(index)

            frame_counter = 0

        if face_locations and face_names:
            for loc, name in zip(face_locations, face_names):
                upd_loc = map(lambda x: x * compression_ratio, list(loc))
                top, right, bottom, left = upd_loc
                cv2.rectangle(flipped_frame, (left, top), (right, bottom), text_color, 3)
                font = cv2.FONT_HERSHEY_TRIPLEX
                cv2.putText(flipped_frame, name, (left + 6, bottom - 6), font, 1.0, frame_color)

        cv2.imshow("Video", flipped_frame)
        key = cv2.waitKey(1)
        if key == exit_key:
            break

    database.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_file: str = sys.argv[0] if len(sys.argv) == 2 else "settings.json"

    with open(config_file) as file:
        settings = json.load(file)
        db_settings = settings["Database"]
        recognize_limiter = settings["Common"]["recognize_counter_limit"]
        remember_limiter = settings["Common"]["remember_counter_limit"]
        compression = settings["Speed"]["compression_ratio"]
        frame_limiter = settings["Speed"]["frame_counter_limit"]
        cam = settings["Common"]["camera_channel"]
        exit_value = settings["Common"]["exit_key"]
        tc = settings["Customize"]["text_color"]
        tcolor = tuple(int(tc[i:i + 2], 16) for i in (1, 3, 5))
        fc = settings["Customize"]["frame_color"]
        fcolor = tuple(int(fc[i:i + 2], 16) for i in (1, 3, 5))

    main(db_settings, recognize_limiter, remember_limiter, compression, frame_limiter, cam, exit_value, tcolor, fcolor)
