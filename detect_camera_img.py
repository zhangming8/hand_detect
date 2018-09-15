#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os, cv2, glob, shutil
#sys.path.append(os.path.join(os.getcwd(),'python/'))
from ctypes import *
import math
import random, json


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/Users/ming/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

float_to_image = lib.float_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

float_to_image = lib.float_transfer_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

def readAnnotations(xml_path):
    import xml.etree.cElementTree as ET

    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')

    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text
        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))

        result.append(class_name)
        result.append(x1)
        result.append(y1)
        result.append(x2)
        result.append(y2)

        results.append(result)
    return results


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect_org(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def detect(net, meta, im, thresh=0.7, hier_thresh=.5, nms=.3):
    num = c_int(0)
    pnum = pointer(num)
    f_img = im.astype('float32')
    data = cast(f_img.ctypes.data, POINTER(c_float))
    im = float_to_image(im.shape[1], im.shape[0], 3, data)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                x = b.x  # 中心点x
                y = b.y  # 中心点y
                w = b.w  # 宽度
                h = b.h  # 高度

                left = int(x - w / 2)
                right = int(x + w / 2)
                top = int(y - h/2)
                bot = int(y + h/2)

                # res.append((meta.names[i], dets[j].prob[i], (left, top, right, bot),dets[j].row,dets[j].col,dets[j].mask_num))
                res.append((meta.names[i], dets[j].prob[i], (left, top, right, bot)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res



if __name__ == "__main__":
    # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    # im = load_image("data/wolf.jpg", 0, 0)
    # meta = load_meta("cfg/imagenet1k.data")
    # r = classify(net, meta, im)
    # print r[:10]
    #    net = load_net("/home/ming.zhang04/darknet2/darknet/cfg/yolov3.cfg", "/home/ming.zhang04/darknet2/darknet/yolov3.weights", 0)
    #    meta = load_meta("/home/ming.zhang04/darknet2/darknet/cfg/coco.data")
    #    r = detect(net, meta, "/home/ming.zhang04/darknet2/darknet/data/dog.jpg")
    camera = 0
    net = load_net("yolov3-tiny.cfg", "yolov3-tiny_210000.weights", 0)
    meta = load_meta("voc.data")
    # img = "/Users/ming/Downloads/dataset/Test/1dc9ef1c-8b24-11e8-8865-17405fff1575-3-25.png"

    result_path = "./result"
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    else:
        os.mkdir(result_path)
    #if not os.path.exists(result_path + '/no_detect'):
    #    os.mkdir(result_path + '/no_detect')
    cap = cv2.VideoCapture(camera)
    # cap.set(3, 1280)
    # cap.set(4, 800)
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    index = 0
    while True:
        ret, im = cap.read()  # 读取一帧的图像
        img = im[:, 285:995]  # 710x720
        cv2.imshow("img", img)
        if cv2.waitKey(66) & 0xFF == ord('q'):
            break
        results = detect(net, meta, img, 0.5, 0.5, 0.45)
        # results = []
        index += 1
        if len(results) == 0:
            print("%d can't detect anything" % index)
           # cv2.imwrite(result_path + '/no_detect/' + str(index) + '.jpg', img)

            continue
	else:
            # plot predicted bounding box
            for result in results:
                print('detcted: ', result)
                cls, conf, x0, y0, x1, y1 = result[0], result[1], result[2][0], result[2][1], result[2][2], result[2][3]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(img, str(cls), (int(x0 + (x1 - x0) * 0.5), int(y0 + (y1 - y0) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体粗细
                cv2.putText(img, str(conf), (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('img', img)
            cv2.imwrite(result_path+'/'+ str(index) + '.jpg', img)
    cap.release()
    cv2.destroyAllWindows()

