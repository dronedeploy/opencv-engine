#!/usr/bin/env python
# -*- coding: utf-8 -*-

# thumbor imaging service - opencv engine
# https://github.com/thumbor/opencv-engine

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2014 globo.com timehome@corp.globo.com
from opencv_engine.tiff_support import TiffMixin, TIFF_FORMATS

try:
    import cv
except ImportError:
    import cv2.cv as cv

from colour import Color

from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment

try:
    from thumbor.ext.filters import _composite
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False

FORMATS = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.png': 'PNG'
}

FORMATS.update(TIFF_FORMATS)


class Engine(BaseEngine, TiffMixin):

    @property
    def image_depth(self):
        if self.image is None:
            return 8
        return cv.GetImage(self.image).depth

    @property
    def image_channels(self):
        if self.image is None:
            return 3
        return self.image.channels

    @classmethod
    def parse_hex_color(cls, color):
        try:
            color = Color(color).get_rgb()
            return tuple(c * 255 for c in reversed(color))
        except Exception:
            return None

    def gen_image(self, size, color_value):
        img0 = cv.CreateImage(size, self.image_depth, self.image_channels)
        if color_value == 'transparent':
            color = (255, 255, 255, 255)
        else:
            color = self.parse_hex_color(color_value)
            if not color:
                raise ValueError('Color %s is not valid.' % color_value)
        cv.Set(img0, color)
        return img0

    def create_image(self, buffer, create_alpha=True):
        self.extension = self.extension or '.tif'
        self.no_data_value = None
        # FIXME: opencv doesn't support gifs, even worse, the library
        # segfaults when trying to decoding a gif. An exception is a
        # less drastic measure.
        try:
            if FORMATS[self.extension] == 'GIF':
                raise ValueError("opencv doesn't support gifs")
        except KeyError:
            pass

        if FORMATS[self.extension] == 'TIFF':
            self.buffer = buffer
            img0 = self.read_tiff(buffer, create_alpha)
        else:
            imagefiledata = cv.CreateMatHeader(1, len(buffer), cv.CV_8UC1)
            cv.SetData(imagefiledata, buffer, len(buffer))
            img0 = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_UNCHANGED)

        if FORMATS[self.extension] == 'JPEG':
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass

        return img0

    @property
    def size(self):
        return cv.GetSize(self.image)

    def normalize(self):
        pass

    def resize(self, width, height):
        dims = (int(round(width, 0)), int(round(height, 0)))
        self.image = cv2.resize(numpy.asarray(self.image), dims, interpolation=cv2.INTER_CUBIC)

    def crop(self, left, top, right, bottom):
        x1, y1 = left, top
        x2, y2 = right, bottom
        self.image = self.image[y1:y2, x1:x2]

    def rotate(self, degrees):
        """ rotates the image by specified number of degrees.
            Uses more effecient flip and transpose for multiples of 90

            Args:
                degrees - degrees to rotate image by (CCW)
        """
        image = numpy.asarray(self.image)
        # number passed to flip corresponds to rotation about: (0) x-axis, (1) y-axis, (-1) both axes
        if degrees == 270:
            transposed = cv2.transpose(image)
            rotated = cv2.flip(transposed, 1)
        elif degrees == 180:
            rotated = cv2.flip(image, -1)
        elif degrees == 90:
            transposed = cv2.transpose(image)
            rotated = cv2.flip(transposed, 0)
        else:
            rotated = self._rotate(image, degrees)

        self.image = cv.fromarray(rotated)

    def _rotate(self, image, degrees):
        """ rotate an image about it's center by an arbitrary number of degrees

            Args:
                image - image to rotate (CvMat array)
                degrees - number of degrees to rotate by (CCW)

            Returns:
                rotated image (numpy array)
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, degrees, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def flip_vertically(self):
        """ flip an image vertically (about x-axis) """
        image = numpy.asarray(self.image)
        self.image = cv.fromarray(cv2.flip(image, 0))

    def flip_horizontally(self):
        """ flip an image horizontally (about y-axis) """
        image = numpy.asarray(self.image)
        self.image = cv.fromarray(cv2.flip(image, 1))

    def read(self, extension=None, quality=None):
        if not extension and FORMATS[self.extension] == 'TIFF':
            # If the image loaded was a tiff, return the buffer created earlier.
            return self.buffer
        else:
            if quality is None:
                quality = self.context.config.QUALITY
            options = None
            self.extension = extension or self.extension

            try:
                if FORMATS[self.extension] == 'JPEG':
                    options = [cv2.IMWRITE_JPEG_QUALITY, quality]
            except KeyError:
                options = [cv2.IMWRITE_JPEG_QUALITY, quality]

            if FORMATS[self.extension] == 'TIFF':
                channels = cv2.split(numpy.asarray(self.image))
                data = self.write_channels_to_tiff_buffer(channels)
            else:
                success, numpy_data = cv2.imencode(self.extension, numpy.asarray(self.image), options or [])
                if success:
                    data = numpy_data.tostring()
                else:
                    raise Exception("Failed to encode image")

            if FORMATS[self.extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
                if hasattr(self, 'exif'):
                    img = JpegFile.fromString(data)
                    img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                    data = img.writeString()

        return data

    def set_image_data(self, data):
        cv.SetData(self.image, data)

    def image_data_as_rgb(self, update_image=True):
        if self.image.channels == 4:
            mode = 'BGRA'
        elif self.image.channels == 3:
            mode = 'BGR'
        else:
            raise NotImplementedError("Only support fetching image data as RGB for 3/4 channel images")
        return mode, self.image.tobytes()
