import cv2
import numpy as np
import os

from non_max_suppression import non_max_suppression_fast as nms

if not os.path.isdir('CarData'):
    print('CarData folder not found')
    exit(1)

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 300

SVM_SCORE_THRESHOLD = 1.8
# maximum acceptable proportion of overlap in the NMS step
NMS_OVERLAP_THRESHOLD = 0.15

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                    trees=5)  # index_kdtree? trees=5?
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)

bow_kmeans_trainer = cv2.BOWKMeansTrainer(12)  # number of clusters
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)


def get_pos_and_neg_paths(i):
    pos_path = 'CarData/TrainImages/pos-%d.pgm' % (i + 1)
    neg_path = 'CarData/TrainImages/neg-%d.pgm' % (i + 1)
    return pos_path, neg_path


def add_sample(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, mask=None)
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors)


def pyramid_generator(gray_img, scale_factor=1.25, min_size=(200, 80), max_size=(600, 600)):
    h, w = gray_img.shape
    min_w, min_h = min_size
    max_w, max_h = max_size
    while w >= min_w and h >= min_h:
        if w <= max_w and h <= max_h:
            yield gray_img
        w /= scale_factor
        h /= scale_factor
        gray_img = cv2.resize(gray_img, (int(w), int(h)),
                              interpolation=cv2.INTER_AREA)


def sliding_window(img, step=20, window_size=(100, 40)):
    img_h, img_w = img.shape
    window_w, window_h = window_size
    for y in range(0, img_w, step):
        for x in range(0, img_h, step):
            roi = img[y:y+window_h, x:x+window_w]
            roi_h, roi_w = roi.shape
            if roi_w == window_w and roi_h == window_h:
                yield (x, y, roi)


for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    add_sample(pos_path)
    add_sample(neg_path)

voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)


def extract_bow_descriptors(img):
    features = sift.detect(img)
    return bow_extractor.compute(img, keypoints=features)


# Import training data and labels
training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_paths(i)
    # insert positive data
    pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    pos_descriptors = extract_bow_descriptors(pos_img)
    if pos_descriptors is not None:
        training_data.extend(pos_descriptors)
        training_labels.append(1)
    # insert negative data
    neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    neg_descriptors = extract_bow_descriptors(neg_img)
    if neg_descriptors is not None:
        training_data.extend(neg_descriptors)
        training_labels.append(-1)

outputModelPath = 'mysvm.xml'
svm = cv2.ml.SVM_create()
if os.path.exists(outputModelPath):
    svm.load(outputModelPath)
else:
    # specifying the classifier's level of strictness
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(50)
    svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
            np.array(training_labels))
    # Export trained model
    svm.save(outputModelPath)

for test_img_path in ['CarData/TestImages/test-0.pgm',
                      'CarData/TestImages/test-1.pgm',
                      '../images/car.jpg',
                      '../images/haying.jpg',
                      '../images/statue.jpg',
                      '../images/woodcutters.jpg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pos_rects = []
    for resized in pyramid_generator(gray_img):
        for x, y, roi in sliding_window(resized):
            descriptors = extract_bow_descriptors(roi)
            if descriptors is None:
                continue
            prediction = svm.predict(descriptors)
            if prediction[1][0][0] == 1.0:
                # return a score as part of its output, lower value
                # represents a higher level of confidence.
                raw_prediction = svm.predict(
                    descriptors, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                score = -raw_prediction[1][0][0]
                if score > SVM_SCORE_THRESHOLD:
                    h, w = roi.shape
                    scale = gray_img.shape[0] / float(resized.shape[0])
                    pos_rects.append(
                        [int(x * scale), int(y * scale), int((x + w) * scale),
                            int((y + h) * scale), score])
    pos_rects = nms(np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
    for x0, y0, x1, y1, score in pos_rects:
        cv2.rectangle(img, (int(x0), int(y0)),
                      (int(x1), int(y1)), (0, 255, 0), 2)
        text = '%.2f' % score
        cv2.putText(img, text, (int(x0), int(y0) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow(test_img_path, img)
cv2.waitKey(0)

