from ultralytics import YOLO
import cv2
import time
import collections
try:
    from collections import abc
    print("READY")
    collections.MutableMapping = abc.MutableMapping
except:
    pass

from dronekit import connect, VehicleMode
from pymavlink import mavutil

# Koneksi ke drone
print('Connecting...')
vehicle = connect('tcp:127.0.0.1:5762')

print("WAITING...")

def arm_and_takeoff(altitude):
    while not vehicle.is_armable:
        print("waiting to be armable")
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    print("Taking Off")
    vehicle.simple_takeoff(altitude)

    while True:
        v_alt = vehicle.location.global_relative_frame.alt
        print(">> Altitude = %.1f m" % v_alt)
        if v_alt >= altitude - 1.0:
            print("Target altitude reached")
            break

def gerak(vx, vy):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, 0,
        0, 0, 0,
        0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

# Load model YOLO
model = YOLO("yolov11.pt")
cap = cv2.VideoCapture(0)  # Ubah ke kamera

prev_time = time.time()  # Waktu sebelumnya untuk menghitung FPS
vx = 2  # Kecepatan maju kapal
vy = 0  # Kecepatan samping kapal

arm_and_takeoff(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Jalankan inferensi YOLO pada frame
    results = model(frame)

    # Tambahkan bounding box ke frame
    annotated_frame = results[0].plot()

    # Hitung dimensi frame
    height, width, _ = annotated_frame.shape

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Tampilkan FPS dan kecepatan di pojok kiri atas
    cv2.putText(annotated_frame, f"FPS: {fps:.2f} Vy: {vy}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Vx: {vx}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Gambar grid 3x4 pada frame dan tambahkan angka
    rows, cols = 4, 3
    box_number = 1
    detected_objects = []
    for i in range(rows):
        for j in range(cols):
            x_start = j * width // cols
            y_start = i * height // rows
            x_end = (j + 1) * width // cols
            y_end = (i + 1) * height // rows

            # Kotak hitam untuk bagian 1-6
            if box_number <= 6:
                cv2.rectangle(annotated_frame, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)

            # Cari objek dalam grid
            for obj in results[0].boxes.data.tolist():
                x_center, y_center = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
                if x_start < x_center < x_end and y_start < y_center < y_end:
                    detected_objects.append((box_number, obj[4], obj[5]))  # (Box, Confidence, Class)

            box_number += 1

    # Filter deteksi
    red_balls = [obj for obj in detected_objects if obj[2] == 0]  # Bola merah
    green_balls = [obj for obj in detected_objects if obj[2] == 1]  # Bola hijau

    # Logika navigasi
    if len(red_balls) > 0 and len(green_balls) > 0:
        vy = 0  # Jika ada dua warna, kapal tetap lurus
    elif len(red_balls) > 0:
        largest_red = max(red_balls, key=lambda x: x[1])  # Bola merah terbesar
        if largest_red[0] in [4, 7]:
            vy = 1
        elif largest_red[0] in [5, 8]:
            vy = 2
    elif len(green_balls) > 0:
        largest_green = max(green_balls, key=lambda x: x[1])  # Bola hijau terbesar
        if largest_green[0] in [6, 9]:
            vy = -1
        elif largest_green[0] in [5, 8]:
            vy = -2
    else:
        vy = 0  # Tidak ada deteksi, tetap lurus

    # Kirim perintah ke drone
    gerak(vx, vy)

    # Tampilkan frame
    cv2.imshow("Deteksi Bola - Video", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua sumber daya
cap.release()
cv2.destroyAllWindows()
vehicle.close()

print("Proses selesai.")
