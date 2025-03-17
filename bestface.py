import cv2
import insightface
from insightface.app import FaceAnalysis

# Khởi tạo ứng dụng InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Đọc video
video_path = r"C:\Users\Admin\Desktop\video\IMG_0391.MP4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
best_frame = None
best_score = float('inf')  # Giá trị nhỏ nhất thể hiện mặt gần chính diện nhất
best_timestamp = 0

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy số khung hình trên giây

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = frame_count / fps  # Tính thời gian hiện tại của video (giây)

    if frame_count % 5 != 0:  # Chỉ kiểm tra mỗi 5 khung hình để tăng tốc
        continue

    faces = app.get(frame)

    for face in faces:
        yaw, pitch, roll = face.pose

        # Chỉ chọn mặt gần chính diện (góc nhỏ)
        if abs(yaw) < 10 and abs(pitch) < 10 and abs(roll) < 10:
            score = abs(yaw) + abs(pitch) + abs(roll)  # Tổng các góc

            if score < best_score:
                best_score = score
                best_frame = frame.copy()
                best_timestamp = timestamp  # Lưu thời gian tốt nhất

cap.release()

# Lưu ảnh của khung hình có mặt chính diện nhất
if best_frame is not None:
    cv2.imwrite("best_frame.jpg", best_frame)
    print(f"Đã lưu ảnh tại thời điểm {best_timestamp:.2f} giây: best_frame.jpg")
else:
    print("Không tìm thấy khuôn mặt chính diện nào.")
