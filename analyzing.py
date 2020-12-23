import base64

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# where the photo file located
img_path = "uploads/asd.png"
#array of black photo
foreground_img = []
# where the object detection config file
detect_cfg = 'models/yolov3_testing.cfg'
# flower weights
flower_weights = 'models/yolov3_training_final.weights'
# leaf weights
leaf_weights = 'models/yolov3_leaves_training_final.weights'
# load the flower model that saved in colab
flower_model = load_model('models/7-flowers-classifier.h5')
# lode the leaf model
leaf_model = load_model('models/4-leaves-classifier.h5')



# gets the photo
def insertImg(img_path):
    img = cv2.imread(img_path)  # image read
    copy_img = cv2.imread(img_path)  # copy image
    # arrays for prediction and image we return
    flowers = []
    leaves = []

    flower_objects, img1 = detectObject(img, flower_weights, class_name='flower')
    leaf_objects, img2 = detectObject(copy_img, leaf_weights, class_name='leaf')


    # recognize flowers objects if found any
    if len(flower_objects) > 0:
        for flower_img in flower_objects:
            cv2.imwrite('uploads/detect_img.jpg', flower_img)

            encoded_string = ""
            with open("uploads/detect_img.jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            fg_img = backgroundDetect(flower_img)
            foreground_img.append(fg_img)
            cv2.imwrite('uploads/crop_img.jpg', fg_img)
            tmp_img = image.load_img('uploads/crop_img.jpg', target_size=(132, 132))
            tmp_img = image.img_to_array(tmp_img)
            tmp_img = np.expand_dims(tmp_img, axis=0)
            p = flower_model.predict(tmp_img)
            prediction_class = np.argmax(p)
            pred_str = pred_flower(prediction_class, p)

            # add to json all the decoded predictions
            js = {"image": encoded_string.decode('utf-8'), "info": pred_str}
            flowers.append(js)


    # recognize leaves objects if found any
    if len(leaf_objects) > 0:
        for leaf_img in leaf_objects:
            cv2.imwrite('uploads/detect1_img.jpg', leaf_img)

            encoded_string = ""
            with open("uploads/detect1_img.jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            fg_img = backgroundDetect(leaf_img)
            foreground_img.append(fg_img)
            cv2.imwrite('uploads/crop_img.jpg', fg_img)
            tmp_img = image.load_img('uploads/crop_img.jpg', target_size=(132, 132))
            tmp_img = image.img_to_array(tmp_img)
            tmp_img = np.expand_dims(tmp_img, axis=0)
            p = leaf_model.predict(tmp_img)
            prediction_class = np.argmax(p)
            pred_str = pred_leaf(prediction_class, p)

            # add to json all the decoded predictions
            js = {"image": encoded_string.decode('utf-8'), "info": pred_str}
            leaves.append(js)
    return {"flowers":flowers,"leaves":leaves}


# red. green . blue
def detectObject(img, weights, class_name):
    # if the class is flower, find them and pick a red square
    if class_name == 'flower':
        color = (255, 0, 0)
    else:
        # if the class is leaf, find them and pick a blue square
        color = (0, 0, 255)
    # read the image to copy
    copy_img = cv2.imread(img_path)
    # Load Yolo
    # create network
    net = cv2.dnn.readNet(weights, detect_cfg)
    small_imgs = []
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    height, width, channels = img.shape
    # Detecting objects
    # change the photo to something that the network can recognize
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # add the photo to the network
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing information on the screen
    confidences = []
    boxes = []
    # loop all te candidate objects to square
    for out in outs:
        for detection in out:
            # give a score to what its look like
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # get the location and size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

        # delete who not get over 0.5
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            x = abs(x)
            y = abs(y)

            small_imgs.append(img[y:y + h, x:x + w, 0:3])
            cv2.rectangle(copy_img, (x, y), (x + w, y + h), color, 2)

    return small_imgs, copy_img


def backgroundDetect(img):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    w, h, ch = img.shape
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (1, 1, w, h)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
    fg_img = img * mask2[:, :, np.newaxis]
    return fg_img


def pred_flower(pred_cls, p):
    if p[0, pred_cls] < 0.5:
        # Changed from 0.7 to 0.5
        str = "the system couldn't recognize the flower"
    else:
        if pred_cls == 0:
            str = "the flower is daisy"
        elif pred_cls == 1:
            str = "the flower is dandelion"
        elif pred_cls == 2:
            str = "the flower is iris"
        elif pred_cls == 3:
            str = "the flower is a rose"
        elif pred_cls == 4:
            str = "the flower is sunflower"
        elif pred_cls == 5:
            str = "the flower is tulip"
        elif pred_cls == 6:
            str = "the flower is water lily"
    return str


def pred_leaf(pred_cls, p):

    if p[0, pred_cls] < 0.5:
        str = "the system couldn't recognize the leaf's health condition"
    else:
        if pred_cls == 0:
            str = "the leaf suffer from bugs"
        elif pred_cls == 1:
            str = "the leaf is dehydrate"
        elif pred_cls == 2:
            str = "the leaf is healthy"
        elif pred_cls == 3:
            str = "the leaf suffer from lack of magnesium"

    return str

