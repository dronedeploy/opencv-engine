#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2016 fanhero.com christian@fanhero.com

import cv2
import numpy as np

from colour import Color
from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment

from opencv_engine.tiff_support import TiffMixin, TIFF_FORMATS

try:
    from thumbor.ext.filters import _composite
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False

FORMATS = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.png': 'PNG',
    '.webp': 'WEBP'
}

FORMATS.update(TIFF_FORMATS)


class Engine(BaseEngine, TiffMixin):
    @property
    def image_depth(self):
        if self.image is None:
            return np.uint8
        return self.image.dtype

    @property
    def image_channels(self):
        if self.image is None:
            return 3
        try:
            return self.image.shape[2]
        except IndexError:
            return 1

    @classmethod
    def parse_hex_color(cls, color):
        try:
            color = Color(color).get_rgb()
            return tuple(c * 255 for c in reversed(color))
        except Exception:
            return None

    def gen_image(self, size, color_value):
        if color_value == 'transparent':
            color = (255, 255, 255, 255)
            img = np.zeros((size[1], size[0], 4), self.image_depth)
        else:
            img = np.zeros((size[1], size[0], self.image_channels), self.image_depth)
            color = self.parse_hex_color(color_value)
            if not color:
                raise ValueError('Color %s is not valid.' % color_value)
        img[:] = color
        return img

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
            img = self.read_tiff(buffer, create_alpha)
        else:
            img = cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)

        if FORMATS[self.extension] == 'JPEG':
            self.exif = None
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass
        return img

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]

    def normalize(self):
        pass

    def resize(self, width, height):
        dims = (int(round(width, 0)), int(round(height, 0)))
        self.image = cv2.resize(np.asarray(self.image), dims, interpolation=cv2.INTER_CUBIC)

    def crop(self, left, top, right, bottom):
        self.image = self.image[top: bottom, left: right]

    def rotate(self, degrees):
        """ rotates the image by specified number of degrees.
            Uses more effecient flip and transpose for multiples of 90

            Args:
                degrees - degrees to rotate image by (CCW)
        """
        image = np.asarray(self.image)
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

        self.image = rotated

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
        self.image = np.flipud(self.image)

    def flip_horizontally(self):
        self.image = np.fliplr(self.image)

    def read(self, extension=None, quality=None):
        if not extension and FORMATS[self.extension] == 'TIFF':
            # If the image loaded was a tiff, return the buffer created earlier.
            return self.buffer

        if quality is None:
            quality = self.context.config.QUALITY

        options = None
        extension = extension or self.extension
        try:
            if FORMATS[extension] == 'JPEG':
                options = [cv2.IMWRITE_JPEG_QUALITY, quality]
        except KeyError:
            # default is JPEG so
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]

        try:
            if FORMATS[extension] == 'WEBP':
                options = [cv2.IMWRITE_WEBP_QUALITY, quality]
        except KeyError:
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]

        if FORMATS[self.extension] == 'TIFF':
            channels = cv2.split(np.asarray(self.image))
            data = self.write_channels_to_tiff_buffer(channels)
        else:
            success, buf = cv2.imencode(extension, self.image, options or [])
            if success:
                data = buf.tostring()
            else:
                raise Exception("Failed to encode image")

        if FORMATS[extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
            if hasattr(self, 'exif') and self.exif != None:
                img = JpegFile.fromString(data)
                img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                data = img.writeString()

        return data

    def set_image_data(self, data):
        self.image = np.frombuffer(data, dtype=self.image.dtype).reshape(self.image.shape)

    def image_data_as_rgb(self, update_image=True):
        if self.image_channels == 4:
            mode = 'BGRA'
        elif self.image_channels == 3:
            mode = 'BGR'
        else:
            raise NotImplementedError("Only support fetching image data as RGB for 3/4 channel images")
        return mode, self.image.tostring()

    def draw_rectangle(self, x, y, width, height):
        cv2.rectangle(self.image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 255, 255))

    def convert_to_grayscale(self, update_image=True, with_alpha=True):
        image = None
        if self.image_channels >= 3 and with_alpha:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        elif self.image_channels >= 3:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif self.image_channels == 1:
            # Already grayscale,
            image = self.image
        if update_image:
            self.image = image
        elif self.image_depth == np.uint16:
            #Feature detector reqiures uint8 images
            image = np.array(image, dtype='uint8')
        return image

    def paste(self, other_engine, pos, merge=True):
        if merge and not FILTERS_AVAILABLE:
            raise RuntimeError(
                'You need filters enabled to use paste with merge. Please reinstall ' +
                'thumbor with proper compilation of its filters.')

        self.enable_alpha()
        other_engine.enable_alpha()

        sz = self.size
        other_size = other_engine.size

        mode, data = self.image_data_as_rgb()
        other_mode, other_data = other_engine.image_data_as_rgb()

        imgdata = _composite.apply(
            mode, data, sz[0], sz[1],
            other_data, other_size[0], other_size[1], pos[0], pos[1], merge)

        self.set_image_data(imgdata)

    def enable_alpha(self):
        if self.image_channels < 4:
            with_alpha = np.zeros((self.size[1], self.size[0], 4), self.image.dtype)
            if self.image_channels == 3:
                cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA, with_alpha)
            else:
                cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGRA, with_alpha)
            self.image = with_alpha
