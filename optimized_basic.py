import json
import os
import sys
import cv2
import face_recognition
import numpy as np
import re
import time


def compare_faces(faces_list: list[np.ndarray], face: np.ndarray, tolerance: float = 0.6) -> (bool, int):
    face_distances = face_recognition.face_distance(faces_list, face)
    match_index = (np.argmin(face_distances) if face_distances.size else -1)
    return (match_index >= 0 and face_distances[match_index] <= tolerance), match_index


def main(compression_ratio: int, frame_counter_limit: int, camera_channel: int, exit_key: int,
         text_color: tuple = (0, 255, 0), frame_color: tuple = (255, 0, 0)) -> None:
    # уходим в базовую реализацию при выставлении compression_ration = 1, frame_counter_limit = 1
    camera = cv2.VideoCapture(camera_channel)

    known_faces = []
    known_faces_name = []
    directory_path = "images/"
    files = [f for f in os.listdir(directory_path)
             if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]
    for f in files:
        filepath = os.path.join(directory_path, f)
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img_rgb)
        if enc:
            measurements = enc[0]
            known_faces.append(measurements)
            name = "".join(f.split(".")[:-1])
            known_faces_name.append(name)

    face_locations = []
    face_names = []
    frame_counter = 0
    while True:
        start_time = time.time()
        result, frame = camera.read()
        flipped_frame = cv2.flip(frame, 1)
        frame_counter += 1

        if frame_counter >= frame_counter_limit:
            small_flipped_frame = cv2.resize(flipped_frame, (0, 0), fx=1/compression_ratio, fy=1/compression_ratio)
            rgb_flipped_frame = cv2.cvtColor(small_flipped_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_flipped_frame)
            face_encodings = face_recognition.face_encodings(rgb_flipped_frame, face_locations)

            face_names = []
            for face_enc in face_encodings:
                res, index = compare_faces(known_faces, face_enc)
                name = "Unknown"
                if res:
                    name = known_faces_name[index]
                face_names.append(name)
            frame_counter = 0

        if face_locations and face_names:
            for loc, name in zip(face_locations, face_names):
                upd_loc = map(lambda x: x * compression_ratio, list(loc))
                top, right, bottom, left = upd_loc
                cv2.rectangle(flipped_frame, (left, top), (right, bottom), text_color, 3)
                font = cv2.FONT_HERSHEY_TRIPLEX
                cv2.putText(flipped_frame, name, (left + 6, bottom - 6),
                            font, 1.0, frame_color)

        cv2.imshow('Face Recognition', flipped_frame)

        end_time = time.time()
        eclipsed_time = end_time - start_time
        print("Time to frame:", eclipsed_time)

        key = cv2.waitKey(1)
        if key == exit_key:
            break

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

    main(compression, frame_limiter, cam, exit_value, tcolor, fcolor)
