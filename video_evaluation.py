from LogisticRegression import LogisticRegressionModel
import torch
import os
import cv2
from torchvision import transforms, utils
import torch.nn as nn
import collections
from PIL import Image
import numpy as np


def generate_fft_input(video_path, model):
    clear_fft_directory()

    i = 0
    predicted_scalar = 0

    vidcap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True
    while success:
        # Frame extraction
        success,image = vidcap.read(0)
        if not success:
            break
        #cv2.imwrite("frames/frame{}.jpg".format(frame_count), image)
        #print ('Read a new frame: ', success)
        # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # Facial detection/extraction
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.85
        )

        # Crop background so that the image is just the face
        j = 0
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = image[ny:ny+nr, nx:nx+nr]
            if len(faceimg) == 0:
                break
            faceimg = cv2.resize(faceimg, (281, 281))
            j += 1
            #cv2.imwrite("frames_faces/frame{}_{}.jpg".format(frameNumber, j), faceimg)

            #FFT algorithm
            f = np.fft.fft2(faceimg)
            fshift = np.fft.fftshift(f)
            # The magnitude size is likely to be a hyper parameter in this case
            magnitude_spectrum = 10*np.log(np.abs(fshift))

            frame_file_name = 'frame{}_{}.jpg'.format(frame_count, j)
            # Create FFT representation of face
            cv2.imwrite('frames_fft_video_evaluation/' + frame_file_name, magnitude_spectrum)
            image = Image.open('frames_fft_video_evaluation/' + frame_file_name)
            image = transforms.ToTensor()(image)
            image = torch.autograd.Variable(image.view(-1, input_dim))
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            predicted_scalar += predicted[0].item()
            i += 1
    if predicted_scalar >= i / 2:
        print("Video is real")
        return 1
    else:
        print("Video is fake")
        return 0


input_dim = 281*281
output_dim = 2
model = LogisticRegressionModel(input_dim, output_dim)
model.load_state_dict(torch.load('model_youtube_15_magnitude.pt'))

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Fake video evaluation
i = 0

youtube_fake_video_path = '/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/manipulated_sequences/DeepFakeDetection/c23/videos/'
youtube_real_video_path = '/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/original_sequences/actors/c23/videos/'

actor_fake_video_path = '/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/manipulated_sequences/Deepfakes/c23/videos/'
actor_real_video_path = '/Volumes/antideepfake/FaceForensics++dataset_larger_compressed/original_sequences/youtube/c23/videos/'


def clear_fft_directory():
    for filename in os.listdir('frames_fft_video_evaluation'):
        os.remove('frames_fft_video_evaluation/' + filename)

for filename in os.listdir(youtube_fake_video_path):
    if i >=6:
        i = 0
        break
    result = generate_fft_input(youtube_fake_video_path + filename, model)
    if result == 0:
        true_negatives += 1
    else:
        false_negatives += 1
    i += 1 

# Real video evaluation
for filename in os.listdir(youtube_real_video_path):
    if i >=6:
        i = 0
        break
    result = generate_fft_input(youtube_real_video_path + filename, model)
    if result == 1:
        true_positives += 1
    else:
        false_positives += 1
    i +=1

print("TP: {} FP: {} TN: {} FN: {}".format(true_positives, false_positives, true_negatives, false_negatives))




