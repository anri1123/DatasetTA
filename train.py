import cv2  # type: ignore
import time
from ultralytics import YOLO  # type: ignore

# Load model hasil training
model_path = 'best.pt'  # Ganti dengan path model hasil trainingmu
model = YOLO(model_path)

# Inisialisasi kamera laptop
cap = cv2.VideoCapture(0)

# Cek apakah kamera berhasil terbuka
if not cap.isOpened():
    print("Tidak dapat membuka kamera laptop.")
    exit()

# Variabel untuk menghitung FPS
prev_time = 0

# Loop untuk membaca frame dari kamera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Dapatkan waktu saat ini untuk menghitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Deteksi objek menggunakan YOLOv8 dengan threshold confidence 0.7
    results = model.predict(frame, conf=0.7)

    # Ambil frame dengan bounding box yang telah ditambahkan
    annotated_frame = results[0].plot()

    # Tambahkan FPS di tampilan
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan hasil deteksi dalam satu jendela saja
    cv2.imshow('Deteksi Sampah dengan YOLOv8', annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
