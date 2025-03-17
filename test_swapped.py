import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# # Tải mô hình hoán đổi khuôn mặt
swapper = insightface.model_zoo.get_model(r'J:\LTSoftware\Project\insightface\web-demos\src_recognition\inswapper_128.onnx', download=False)

# Khởi tạo ứng dụng nhận diện khuôn mặt
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Đọc ảnh nguồn và ảnh đích
source_img = cv2.imread("source.jpg")  # Ảnh chứa khuôn mặt bạn muốn dùng
target_img = cv2.imread("target.jpg")  # Ảnh chứa khuôn mặt cần thay thế

# Phát hiện khuôn mặt trên cả hai ảnh
source_faces = app.get(source_img)
target_faces = app.get(target_img)

if len(source_faces) == 0 or len(target_faces) == 0:
    print("Không tìm thấy khuôn mặt trong một trong hai ảnh.")
else:
    # Chọn khuôn mặt đầu tiên từ ảnh nguồn và ảnh đích
    source_face = source_faces[0]
    target_face = target_faces[0]

    # Thực hiện hoán đổi khuôn mặt
    swapped_img = swapper.get(target_img, target_face, source_face, paste_back=True)

    # Lưu hoặc hiển thị ảnh kết quả
    cv2.imwrite("swapped.jpg", swapped_img)
    cv2.imshow("Face Swapped", swapped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
