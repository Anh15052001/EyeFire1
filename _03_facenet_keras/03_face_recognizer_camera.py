from scipy.spatial.distance import cosine
import mtcnn
from keras.models import load_model
from utils import *
import time
import imutils
def variance_of_Laplacian(image, size=60):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # calculate detect blur by fft
    fft = np.fft.fft2(image)
    # shift the zero frequency component
    fftShift = np.fft.fftshift(fft)
    # check is vis = True will visualize

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
def recognize(img,
              detector,
              encoder,
              encoding_dict,
              recognition_t=0.5,
              confidence_t=0.99,
              required_size=(160, 160), ):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img


if __name__ == '__main__':
    encoder_model = 'data/model/facenet_keras.h5'
    encodings_path = 'data/encodings/encodings.pkl'

    face_detector = mtcnn.MTCNN()
    face_encoder = load_model(encoder_model)
    encoding_dict = load_pickle(encodings_path)

    vc = cv2.VideoCapture(0)
    while vc.isOpened():
        start_time = time.time()
        ret, frame = vc.read()
        if not ret:
            print("no frame:(")
            break
        orig = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        thresh = variance_of_Laplacian(gray)
        if thresh < 10:
            cv2.putText(frame, "Too blur", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            frame = recognize(frame, face_detector, face_encoder, encoding_dict)
            end_time = time.time()
            print("During time: ", end_time-start_time)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
