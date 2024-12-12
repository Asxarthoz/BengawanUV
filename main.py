from ultralytics import YOLO
import cv2
import time

# Load model YOLO menggunakan file 'yolov5.pt'
model = YOLO("yolov11.pt")

# Ganti '0' dengan path file video Anda
cap = cv2.VideoCapture("vidioPOV1.mp4")  # Ganti dengan path ke file video Anda

prev_time = time.time()  # Waktu sebelumnya untuk menghitung FPS

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

    # Tampilkan FPS di pojok kiri atas
    cv2.putText(
        annotated_frame, 
        f"FPS: {fps:.2f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )

    # Gambar grid 3x4 pada frame
    rows, cols = 4, 3  # 4 baris, 3 kolom
    for i in range(1, rows):
        cv2.line(annotated_frame, (0, i * height // rows), (width, i * height // rows), (255, 0, 0), 2)
    for j in range(1, cols):
        cv2.line(annotated_frame, (j * width // cols, 0), (j * width // cols, height), (255, 0, 0), 2)

    # Tampilkan frame yang dianotasi di window baru
    cv2.imshow("Deteksi Bola - Video", annotated_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua sumber daya
cap.release()
cv2.destroyAllWindows()

print("Proses selesai.")
