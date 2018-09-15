from scipy import io
from xml_writer import PascalVocWriter
import glob
import cv2
import os
import shutil
import numpy as np

mat_path = './annotations'
img_path = './images'
save_path = './hand_dataset_xml'
files = glob.glob(mat_path + '/*.mat')
img_files = glob.glob(img_path + '/*.jpg')
assert len(files) == len(img_files)
for f in files:
    mat = io.loadmat(f)
    box = mat['boxes'][0]
    print f
    # print '----------'
    # print box
    img = cv2.imread(img_path + '/' + os.path.basename(f)[:-4] + '.jpg')
    if type(img) == type(None):
        print('images %s: read error' %(f))
        continue
    img_h, img_w, _ = img.shape
    writer = PascalVocWriter('./', os.path.basename(f)[:-4] + '.jpg', (img_h, img_w, 3), localImgPath='./',
                             usrname="hand_dataset")
    writer.verified = True
    for b in box:
        x, y = [], []
        for i in b[0][0]:
            if len(x) == 4:
                continue
            # print "------------"
            # print i
            x.append(int(i[0][1]))
            y.append(int(i[0][0]))
        writer.addBndBox(np.min(x), np.min(y), np.max(x), np.max(y), 'hand', 0)
    writer.save(targetFile=save_path + '/' + os.path.basename(f)[:-4] + '.xml')
    shutil.copyfile(img_path + '/' + os.path.basename(f)[:-4] + '.jpg',
                    save_path + '/' + os.path.basename(f)[:-4] + '.jpg')

