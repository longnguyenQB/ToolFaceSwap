import cv2
import insightface
import os
import json
from insightface.app import FaceAnalysis

def read_js(path):
    f = open(path, "r", encoding='utf-8')
    data = json.loads(f.read())
    f.close()
    return data


def write_js(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file)
    file.close()
    
def get_video_aspect_ratio(video_path):
    """Lấy kích thước và tỉ lệ video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ratio = width / height  # Tính tỉ lệ khung hình
    return width, height, ratio

def crop_to_9_16(video_path, output_path):
    """Cắt video 4:3 để thành 9:16"""
    width, height, ratio = get_video_aspect_ratio(video_path=video_path)
    print("ratio: ", ratio)
    if 0.7 < ratio < 1:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Không thể mở video.")
            return video_path

        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tính toán chiều rộng mới để có tỷ lệ 9:16
        new_width = int(height * (9 / 16))

        if new_width >= width:
            print("Video không thể cắt để thành 9:16, bỏ qua.")
            cap.release()
            return video_path

        left_crop = (width - new_width) // 2  # Cắt đều hai bên

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Cắt khung hình
            cropped_frame = frame[:, left_crop:left_crop + new_width]

            # Ghi frame đã cắt
            out.write(cropped_frame)

        cap.release()
        out.release()

        try:
            os.remove(video_path)
        except:
            pass
        print(f"Đã cắt video 4:3 thành 9:16 và lưu tại: {output_path}")
        return output_path
    else:
        return video_path

def check_face(path_img):
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    # Mở ảnh
    try:
        img = cv2.imread(path_img)
        faces = app.get(img)  # Trích xuất khuôn mặt
    except Exception as e:
        print(f"Không thể mở ảnh {path_img}: {e}")
        return False

    if len(faces) > 0:
        return True
    else:
        return False

def get_bestface(video_path, output_path):
    # Khởi tạo ứng dụng InsightFace
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Đọc video
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
        cv2.imwrite(output_path, best_frame)
        print(f"Đã lưu ảnh tại thời điểm {best_timestamp:.2f} giây: {output_path}")
        return output_path
    else:
        print("Không tìm thấy khuôn mặt chính diện nào.")
        return None
    
def convert_path(path):
    path = path.replace("\a", "//a")
    path = path.replace("\t", "//t")
    path = path.replace("\r", "//r")
    path = path.replace("\n", "//n")
    path = path.replace("\f", "//f")
    path = path.replace("\v", "//v")
    path = path.replace("\b", "//b")
    return path