# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from libs.pascal_voc_io import PascalVocWriter
import os.path
import sys
import json


class LabelFileError(Exception):
    pass


class LabelFile(object):
    # It might be changed as window creates
    suffix = '.lif'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False
        self.verified_user = None
        self.verified_time = None

    def saveJsonFormat(self, filename, shapes, imagePath, imageData=None,
                        lineColor=None, fillColor=None, databaseSrc=None):
        JSON_EXT = '.json'
        save_json = {}
        imgFileName = os.path.basename(imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        image = QImage()
        image.load(imagePath)
        imageShape = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]
        save_json['filename'] = imgFileNameWithoutExt
        save_json['height'] = imageShape[0]
        save_json['width'] = imageShape[1]
        save_json['depth'] = imageShape[2]
        save_json['verified'] = self.verified
        save_json['verified_user'] = self.verified_user
        save_json['verified_time'] = self.verified_time
        save_json['object'] = []
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            label_user = shape['label_user']
            label_time = shape['label_time']
            # Add Chris
            difficult = int(shape['difficult'])
            bndbox = LabelFile.convertPoints2BndBox(points)
            single_box = {'name': label, 'xmin': bndbox[0], 'ymin': bndbox[1], 'xmax': bndbox[2], 'ymax': bndbox[3],
                          'label_user': label_user, 'label_time': label_time}
            save_json['object'].append(single_box)
            # writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_json, f, indent=4, ensure_ascii=False)


    def savePascalVocFormat(self, filename, shapes, imagePath, imageData,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = QImage()
        image.load(imagePath)
        imageShape = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            # Add Chris 
            difficult = int(shape['difficult'])
            bndbox = LabelFile.convertPoints2BndBox(points)
            writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult)

        writer.save(targetFile=filename)
        return

    def toggleVerify(self, verified_user, verified_time):
        self.verified = not self.verified
        self.verified_user = verified_user
        self.verified_time = verified_time

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))
