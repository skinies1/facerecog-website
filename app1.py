import streamlit as st
from PIL import Image
import cv2,numpy as np
import dlib
import pandas as pd
import pickle
import face_recognition



face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# FACE_DESC = []
# FACE_NAME = []

def main():
    # Header of page
    st.header('Face Recognition Register')
    st.text_input('Enter Your Student ID')
    st.text_input('Enter Your Name')
    uploaded_file = st.file_uploader('Face uploader')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    uploaded_card = st.file_uploader('card uploader')
    if uploaded_card is not None:
        card = Image.open(uploaded_card)

    if st.button('Train'):
        train(image)
        st.write('image trained')
        train(card)
        st.write("card trained")

    if st.button("compare"):

        image1 = np.array(image)
        encoding1 = face_recognition.face_encodings(image1)[0]

        image2 = np.array(card)
        encoding2 = face_recognition.face_encodings(image2)[0]

        results = face_recognition.compare_faces([encoding1], encoding2)
        persamaan = face_recognition.face_distance([encoding1], encoding2)

        if results[0] == True:
            st.write("Same" , (1.0 - persamaan[0]) *100)



def recog():
    st.header('Face Recognition')
    run = st.checkbox('Run')
    if run:
        run_cam(run)


def run_cam(run):
    cam = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    FACE_DESC, FACE_NAME = pickle.load(open('trainset1.pk', 'rb'))

    while run:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FRAME_WINDOW.image(frame)

        # _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        # new_frame_time = time.time()

        for (x, y, w, h) in faces:
            img = frame[y - 10:y + h + 10, x - 10:x + w + 10][:, :, ::-1]
            dets = detector(img, 1)

            for k, d in enumerate(dets):
                shape = sp(img, d)
                face_desc0 = model.compute_face_descriptor(img, shape, 1)
                d = []
                for face_desc in FACE_DESC:
                    d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
                d = np.array(d)
                idx = np.argmin(d)

                if d[idx] < 0.5:
                    name = FACE_NAME[idx]
                    print(name)
                    cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                    print("Unknow", d[idx])
                    cv2.putText(frame, "Unknow", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

def train(image):
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Face Description...")
    img = np.array(image.convert('RGB'))
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(grey, 1)

    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_desc = model.compute_face_descriptor(img, shape, 10)
        # if face_desc is not None:
        #     st.write(face_desc)
        # else:
        #     st.write("no face found")


    # pickle.dump((face_desc, "TEST PICKLE5"), open('trainp.p','at'))


if __name__ == '__main__':
    st.page_select = st.sidebar.radio('Pages', ['Register','Recognition'])

    if st.page_select == 'Register':
        main()


    if st.page_select == 'Recognition':
        recog()

#แปะ
# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade Pillow