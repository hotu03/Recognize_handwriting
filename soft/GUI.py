
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import string
import os

# Tải mô hình đã lưu ở định dạng .h5
cur_path = os.path.dirname(__file__);
file_path = os.path.join(cur_path, 'mymodel.h5')
if os.path.exists(file_path):
    model = tf.keras.models.load_model(file_path)
    # Biên dịch lại mô hình để tránh cảnh báo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    print(f"File not found: {file_path}")

# Hàm giải mã nhãn thành chữ cái
def decoder_label(arr):
    alphabet = list(string.ascii_lowercase)
    dict_label = {i: alphabet[i] for i in range(26)}
    decoded_arr = [dict_label[i] for i in arr]
    decoded_string = ''.join(decoded_arr)
    return decoded_string

def x_cord_contour(contours):
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return int(M['m10']/M['m00'])
    else:
        pass


def crop_define(path):
    img = cv2.imread(path)
    img1 = cv2.imread(path, 0)
    resized_img = cv2.resize(img1, (4250, 3450))
    crop_img = resized_img[1000:2500, 250:4000]
    reco_letters(crop_img)

def reco_letters(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 800:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)
    preprocessed_letter = []
    for (i, c) in enumerate(contours_left_to_right):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 0), thickness=2)
        digit = thresh[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        preprocessed_letter.append(padded_digit)
    processed_letter = []
    for digit in preprocessed_letter:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))
        processed_letter.append(np.argmax(prediction))
    arr = np.array(processed_letter)
    decoded_string = decoder_label(arr)
    result_text.set(f"Predicted Text: {decoded_string}")
    
    # Cập nhật hình ảnh trong GUI
    img_display = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_display = img_display.resize((425, 345))
    img_display = ImageTk.PhotoImage(img_display)
    panel_image.configure(image=img_display)
    panel_image.image = img_display

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('L')
        img = img.resize((425, 345))
        img = ImageTk.PhotoImage(img)
        panel_image.configure(image=img)
        panel_image.image = img
        crop_define(file_path)

# Tạo giao diện chính
root = tk.Tk()
root.title("Handwritten Character Recognition")
#root.attributes('-fullscreen', True)

# Khung hình ảnh lớn
canvas = tk.Canvas(root, width=425, height=345)
canvas.pack()

panel_image = tk.Label(canvas)
panel_image.pack()

load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Khung hiển thị kết quả
result_frame = tk.Frame(root, bd=2, relief="solid")
result_frame.pack(pady=20)
result_text = tk.StringVar()
result_label = tk.Label(result_frame, textvariable=result_text, font=("Helvetica", 20, "bold"))
result_label.pack()

root.mainloop()

