from imutils.video import VideoStream
from keras.layers import Activation, Input, Dropout, Flatten
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import argparse
import imutils
import time
import cv2
import math
import os
import numpy as np


# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


def train_model():
    def recall(y_true, y_pred):
        """
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.

        Source
        ------
        https://github.com/fchollet/keras/issues/5400#issuecomment-314747992
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1(y_true, y_pred):
        """Calculate the F1 score."""
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r))

    img_width, img_height = 32, 32

    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 212
    nb_validation_samples = 13
    epochs = 5
    batch_size = 4

    save_dir = os.path.join(os.getcwd())
    model_name = '{epoch:03d}.h5'
    filepath = os.path.join(save_dir, model_name)

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # build the VGG16 network
    m = applications.VGG16(
        weights='imagenet', include_top=False, input_shape=input_shape)
    print('Model loaded.')
    print(m.summary())
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Sequential()
    model.add(m)

    model.add(top_model)

    for layer in model.layers[:25]:
        layer.trainable = True

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    callbacks = [checkpoint]

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=["accuracy", f1, recall, precision])
    print(model.summary())

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        callbacks=callbacks,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    return model


def detect_object(image, c=0.5, threshold=0.3):

    res = []
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # load our input image and grab its spatial dimensions
    # image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > c:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, c,
                            threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            res.append((((x, y), (w, h)), LABELS[classIDs[i]]))

    return res


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


# load weights into new model
# loaded_model = get_model()
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
# model = train_model()


trackers = cv2.MultiTracker_create()


if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)


else:
    vs = cv2.VideoCapture(args["video"])


counter = 0

while True:

    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)

    if counter % 5 == 0:
        objects = detect_object(frame)
        team1 = []
        team2 = []
        print(type(frame))
        for i in objects:
            print(i)
            # cv2.rectangle(frame, (i[0][0][0], i[0][0][1]), (i[0][0]
            #                                                 [0] + i[0][1][0], i[0][0][1] + i[0][1][1]), (0, 0, 255), 2)
            # tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            # b = (int(i[0][0][0] + i[0][1][0]), int(i[0][0][1] +
            #                                        i[0][1][1]), int(i[0][0][0]), int(i[0][0][1]))
            # trackers.add(tracker, frame, b)
            if (i[1] == 'person'):

                # cv2.imshow("Frame", frame[i[0][0][1]:i[0][0][1] +
                #                           i[0][1][1], i[0][0][0]:i[0][0][0] + i[0][1][0]])
                # break
                croped_img = np.array(
                    frame[i[0][0][1]:i[0][0][1] + i[0][1][1], i[0][0][0]:i[0][0][0] + i[0][1][0], :])
                print(croped_img.shape)
                # croped_img = np.resize(croped_img, (32, 32,3))
                # croped_img = croped_img.reshape((1, 32, 32, 3))
                # label = model.predict(croped_img)
                cv2.imshow("frame", croped_img)

                # cv2.imshow(str(label), np.array(frame[i[0][0][1]:i[0][0][1] +
                #                           i[0][1][1], i[0][0][0]:i[0][0][0] + i[0][1][0]]))
            elif (i[1] == 'sports ball'):
                pass
            # print(i[1])

    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    counter += 1

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
