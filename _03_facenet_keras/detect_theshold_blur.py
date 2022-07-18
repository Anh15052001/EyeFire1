from scipy.spatial.distance import cosine
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
from utils import get_face, plt_show, get_encode, load_pickle, l2_normalizer
import os
import imutils
import matplotlib.pyplot as plt
def do_blur(img, k_size):
    blur_img = cv2.blur(img, (k_size, k_size))
    return blur_img
def variance_of_Laplacian(image, size=60, vis=False):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # calculate detect blur by fft
    fft = np.fft.fft2(image)
    # shift the zero frequency component
    fftShift = np.fft.fftshift(fft)
    # check is vis = True will visualize
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))
        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean
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
                orig = imutils.resize(img1, width=500)
                gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                val = variance_of_Laplacian(gray, size=60, vis=False)
                value_blur.append(val)
    if flag == True:
        print("Blur: ", k_blur)
        print("Value blur being accept: ", max(value_blur))
        break
