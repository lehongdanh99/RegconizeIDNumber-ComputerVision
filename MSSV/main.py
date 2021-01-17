import sys
import cv2 as cv
import numpy as np
import pytesseract
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog
from PyQt5.QtCore import Qt
from  PyQt5 import uic
from PyQt5.QtGui import QPixmap,QImage
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        # load giao dien
        self.ui = uic.loadUi("GUI_mssv.ui", self)
        # load file weights va file configure
        self.net = cv.dnn.readNetFromDarknet('detectyolo.cfg', 'yolov4-custom_last.weights')
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # hàm sự kiện
        self.ui.btn_loadImage.clicked.connect(self.loadImageClicked)
        self.ui.btn_detect.clicked.connect(self.detect)

    def loadImageClicked(self):
        # load ảnh
        self.fName, _ = QFileDialog.getOpenFileName(self, 'Open file', 'E:\\HK1-2021\\Student Card',
                                                    "Image files (*.jpg *.png)")
        int_img = cv.imread(self.fName)
        if int_img.shape[0] > int_img.shape[1]:
            int_img = cv.rotate(int_img, cv.cv2.ROTATE_90_CLOCKWISE)

        self.img = int_img.copy()
        # convert qua dạng ảnh của qt5
        rgbImage = cv.cvtColor(int_img, cv.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],3*rgbImage.shape[1], QImage.Format_RGB888)
        img = convertToQtFormat.scaled(720, 500, Qt.IgnoreAspectRatio)
        self.ui.lbl_image.setPixmap(QPixmap(img))

    def detect(self):
        self.img = cv.resize(self.img, (416,416), interpolation = cv.INTER_AREA)
        blob = cv.dnn.blobFromImage(self.img, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)
        self.post_process(self.img, outputs, 0.2)
        # blobb = blob.reshape(blob.shape[2],blob.shape[3],1)
        # cv.imshow('Blob',blobb)
             
    def post_process(self, img, outputs, conf):
        H, W = img.shape[:2]

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                # classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0]-2, boxes[i][1])
                (w, h) = (boxes[i][2] + 10, boxes[i][3] + 10)
                img = cv.resize(img, (416,416), interpolation = cv.INTER_AREA)
                cv.imshow("adad", img)
                crop_img = img[y + int(h/3): y + h, x: x + w].copy()
                cv.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 5)
                mssv = pytesseract.image_to_string(crop_img, config="--psm 8")
                lst = list(mssv)
                if lst[0] == "4":
                    lst[0] = "1"
                mssv = ''.join(lst)
                self.ui.lbl_MSSV.setText(mssv[:8])
                
                
                # cv.imshow("agfag",crop_img)
                crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)
                crop_img = QImage(crop_img.data, crop_img.shape[1], crop_img.shape[0], 3*crop_img.shape[1], QImage.Format_RGB888)
                self.ui.lbl_crop.setPixmap(QPixmap.fromImage(crop_img))


if __name__ == "__main__":
    # main(sys.argv[1:])
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())