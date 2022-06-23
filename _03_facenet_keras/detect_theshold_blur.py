from scipy.spatial.distance import cosine
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
from utils import get_face, plt_show, get_encode, load_pickle, l2_normalizer
import os
def do_blur(img, k_size):
    blur_img = cv2.blur(img, (k_size, k_size))
    return blur_img
def variance_of_Laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
encoder_model = 'data/model/facenet_keras.h5'
people_dir = 'data/people'
encodings_path = 'data/encodings/encodings.pkl'
#test_res_path = 'data/results/friends.jpg'
root_img = 'data/gen_test/'
recognition_t = 0.3
required_size = (160, 160)

encoding_dict = load_pickle(encodings_path)
face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)
k_blur = 20
while True:
    flag = True
    value_blur = []
    for path in os.listdir(root_img):
        test_img_path = os.path.join(root_img, path)
        print("Solve: ", test_img_path)
        img = cv2.imread(test_img_path)
        img1 = do_blur(img, k_blur)
        img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        results = face_detector.detect_faces(img_rgb)
        for res in results:
            face, pt_1, pt_2 = get_face(img_rgb, res['box'])
            encode = get_encode(face_encoder, face, required_size)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

            name = 'unknown'
            distance = float("inf")

            for db_name, db_encode in encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < recognition_t and dist < distance:
                    name = db_name
                    distance = dist
            if name == 'unknown':
                flag = False
                k_blur-=1
                break
            else:
                val = variance_of_Laplacian(img1)
                value_blur.append(val)
    if flag == True:
        print("Blur: ", k_blur)
        print("Value blur being accept: ", max(value_blur))
        break
