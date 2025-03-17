from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QSpinBox, QTableWidget, QTableWidgetItem, QFileDialog, QGridLayout,
)
from PySide6.QtCore import *
from PySide6.QtGui import QIcon

import sys
import random
import os
import shutil
from utils import *
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

if not os.path.exists('config'):
    os.makedirs('config')

class BatchProcessor(QWidget):
    def __init__(self):
        super().__init__()

        try:
            configs = read_js(path="./config/configs.json")
        except:
            pass
        self.setWindowTitle("LT_ToolFaceSwap_V0.1")
        self.setGeometry(100, 100, 800, 400)
        # Đặt màu nền đen cho toàn bộ ứng dụng
        self.setStyleSheet("""
            QMainWindow {
                background-color: black;
            }
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        
        layout = QVBoxLayout()

        gridLayout = QGridLayout()
        gridLayout.setObjectName("gridLayout")
        layout.addLayout(gridLayout)

        # Input fields
        self.label = QLabel()
        self.label.setObjectName("label")
        self.label.setText("Đường dẫn thư mục chứa các CCCD:")
        gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.path_cccd_edit = QLineEdit()
        try:
            self.path_cccd_edit.setText(configs["path_cccds"])
        except:
            pass
        gridLayout.addWidget(self.path_cccd_edit, 0, 1, 1, 1)

        self.browse_cccd_btn = QPushButton("...")
        gridLayout.addWidget(self.browse_cccd_btn, 0, 2, 1, 1)


        self.label = QLabel()
        self.label.setObjectName("label")
        self.label.setText("Đường dẫn video:")
        gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.path_video_people_edit = QLineEdit()
        try:
            self.path_video_people_edit.setText(configs["path_video_peoples"])
        except:
            pass
        gridLayout.addWidget(self.path_video_people_edit, 1, 1, 1, 1)

        self.browse_people_btn = QPushButton("...")
        gridLayout.addWidget(self.browse_people_btn, 1, 2, 1, 1)

        self.label = QLabel()
        self.label.setObjectName("label")
        self.label.setText("Thư mục lưu:")
        gridLayout.addWidget(self.label, 2, 0, 1, 1)

        self.path_save_edit = QLineEdit()
        try:
            self.path_save_edit.setText(configs["path_save"])
        except:
            pass
        gridLayout.addWidget(self.path_save_edit, 2, 1, 1, 1)

        self.browse_save_btn = QPushButton("...")
        gridLayout.addWidget(self.browse_save_btn, 2, 2, 1, 1)
        

        
        self.btn_run = QPushButton()
        self.btn_run.setText("Chạy")
        gridLayout.addWidget(self.btn_run, 3, 0, 1, 1)

        self.btn_stop = QPushButton()
        self.btn_stop.setText("Dừng")
        gridLayout.addWidget(self.btn_stop, 4, 0, 1, 1)

        self.label_success = QLabel()
        self.label_success.setObjectName("label_status")
        self.label_success.setText("Thành công: ")
        gridLayout.addWidget(self.label_success, 5, 0, 1, 3)

        self.label_status = QLabel()
        self.label_status.setObjectName("label_status")
        self.label_status.setText("Trạng thái:")
        gridLayout.addWidget(self.label_status, 6, 0, 1, 3)
        
        
        self.setLayout(layout)
        
        # Connect buttons
        self.browse_cccd_btn.clicked.connect(lambda: self.browse_folder(self.path_cccd_edit))
        self.browse_people_btn.clicked.connect(lambda: self.browse_folder(self.path_video_people_edit))
        self.browse_save_btn.clicked.connect(lambda: self.browse_folder(self.path_save_edit))
        self.btn_run.clicked.connect(self.run)
        self.threads = {}
        self.IDs = {}

    def run(self):
        
        path_cccd = self.path_cccd_edit.text()
        path_video_people = self.path_video_people_edit.text()
        path_save = self.path_save_edit.text()
        # number_thread = self.threads_spinbox.value()
        number_thread = 1
        for index_thread in range(number_thread):
            random_bullshit = random.choice(range(1,9999999999))
            self.threads[random_bullshit] = ThreadsSwap(
                                                                path_cccd=path_cccd,
                                                                path_video_people=path_video_people,
                                                                path_save=path_save
                                                                )
            self.threads[random_bullshit].start()
            self.threads[random_bullshit].signal_status.connect(self.update_status)
            self.threads[random_bullshit].signal_success.connect(self.update_success)

    def update_status(self, status):
        self.label_status.setText(status)
    
    def update_success(self, success):
        self.label_success.setText(success)

    def browse_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file")
        if file_path:
            line_edit.setText(file_path)
    
    def browse_folder(self, line_edit):
        folder_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        if folder_path:
            line_edit.setText(folder_path)

class ThreadsSwap(QThread):
    signal_status = Signal(object)
    signal_success = Signal(object)

    def __init__(self,
                 path_cccd,
                 path_video_people,
                 path_save
                 ):

        super(ThreadsSwap, self).__init__()
        self.path_cccd = path_cccd
        self.path_video_people = path_video_people
        self.path_save = path_save
        self.is_running = True

    def run(self):
        self.path_cccds = convert_path(self.path_cccd)
        self.path_video_peoples = convert_path(self.path_video_people)
        self.path_save = convert_path(self.path_save)
        configs = {
                "path_cccds": self.path_cccds,
                "path_video_peoples": self.path_video_peoples,
                "path_save": self.path_save
                   }
        write_js(data=configs, path="./config/configs.json")
        if (not os.path.exists(self.path_cccd)) | (not os.path.exists(self.path_video_people)) | (not os.path.exists(self.path_save)):
            self.signal_status.emit("Đường dẫn không tồn tại. Kiểm tra đường dẫn!")
            self.is_running = False
            self.terminate()
            return
        list_folder_cccd = os.listdir(self.path_cccds)
        list_video_people = os.listdir(self.path_video_peoples)

        # # Tải mô hình hoán đổi khuôn mặt
        swapper = insightface.model_zoo.get_model('./inswapper_128.onnx', download=False)

        # Khởi tạo ứng dụng nhận diện khuôn mặt
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))

        success = 0
        while True:
            try:
                name_folder_cccd = list_folder_cccd.pop(0)
                path_img_front_cccd = os.path.join(self.path_cccds, name_folder_cccd, "CMT_TRUOC.jpg")

                name_video_people = list_video_people.pop(0)
                path_video_people = os.path.join(self.path_video_peoples, name_video_people)
                
                # Convert thành 9:16
                self.signal_status.emit("Đang kiểm tra size và cắt video thành 9:16...")
                path_video_people = crop_to_9_16(video_path=path_video_people, output_path=os.path.join(self.path_video_peoples, "converted_"+name_video_people))
                path_video_people = convert_path(path_video_people)

                # Cắt ảnh mặt chính diện
                self.signal_status.emit("Đang cắt ảnh mặt chính diện...")
                path_img_face = get_bestface(video_path=path_video_people, output_path=os.path.join(self.path_video_peoples, "best_face.jpg"))
                if path_img_face:
                    self.signal_status.emit("Đang hoán đổi mặt...")
                    # Đọc ảnh nguồn và ảnh đích
                    source_img = cv2.imread(path_img_face)  # Ảnh chứa khuôn mặt bạn muốn dùng
                    target_img = cv2.imread(path_img_front_cccd)  # Ảnh chứa khuôn mặt cần thay thế

                    # Phát hiện khuôn mặt trên cả hai ảnh
                    source_faces = app.get(source_img)
                    target_faces = app.get(target_img)

                    if len(source_faces) == 0 or len(target_faces) == 0:
                        print("Không tìm thấy khuôn mặt trong một trong hai ảnh.")
                        try:
                            os.remove(path_img_face)
                        except:
                            pass

                        try:
                            os.remove(path_video_people)
                        except:
                            pass
                    else:
                        # Chọn khuôn mặt đầu tiên từ ảnh nguồn và ảnh đích
                        source_face = source_faces[0]
                        target_face = target_faces[0]

                        # Thực hiện hoán đổi khuôn mặt
                        swapped_img = swapper.get(target_img, target_face, source_face, paste_back=True)

                        # Lưu kết quả

                        ## Tạo thư mục 
                        if not os.path.exists(os.path.join(self.path_save, name_video_people)):
                            os.makedirs(os.path.join(self.path_save, name_video_people))
                        
                        # Copy ảnh cắt vào
                        try:
                            shutil.move(path_img_face, os.path.join(self.path_save, name_video_people))
                        except:
                            pass

                        
                        # Copy video
                        try:
                            shutil.move(path_video_people, os.path.join(self.path_save, name_video_people))
                        except:
                            pass

                        # Copy mặt sau
                        try:
                            shutil.move(os.path.join(self.path_cccds, name_folder_cccd, "CMT_SAU.jpg"), os.path.join(self.path_save, name_video_people))
                        except:
                            pass

                        try:
                            shutil.rmtree(os.path.join(self.path_cccds, name_folder_cccd))
                        except:
                            pass

                        # Lưu hoặc hiển thị ảnh kết quả
                        cv2.imwrite(os.path.join(self.path_save, name_video_people, "CMT_TRUOC.jpg"), swapped_img)
                        # cv2.imshow("Face Swapped", swapped_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        success += 1
                        self.signal_success.emit(f"Thành công: {success}")
            except:
                self.signal_status.emit("Đã chạy xong!")
                break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("logo.ico"))
    window = BatchProcessor()
    window.show()
    sys.exit(app.exec())
