#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment

__version__ = '1.0.1'

try:
    from opencv_engine.engine_cv3 import Engine  # NOQA
except ImportError:
    logging.exception('Could not import opencv_engine. Probably due to setup.py installing it.')

logger = logging.getLogger(__name__)


def _patch_mime_types():
    # need to monkey patch the BaseEngine.get_mimetype function to handle tiffs
    # has to be patched this way b/c called as both a classmethod and instance method internally in thumbor
    old_mime = BaseEngine.get_mimetype

    def new_mime(buffer):
        ''' determine the mime type from the raw image data
            Args:
             buffer - raw image data
            Returns:
             mime - mime type of image
        '''
        mime = old_mime(buffer)
        # tif files start with 'II'
        if not mime and buffer.startswith('II'):
            mime = 'image/tiff'
        return mime

    BaseEngine.get_mimetype = staticmethod(new_mime)


def _patch_exif():
    def _get_exif_segment(self):
        """ Override because the superclass doesn't check for no exif.
        """
        segment = None
        try:
            if getattr(self, 'exif', None) is not None:
                segment = ExifSegment(None, None, self.exif, 'ro')
        except Exception:
            logger.warning('Ignored error handling exif for reorientation', exc_info=True)
        return segment
    BaseEngine._get_exif_segment = _get_exif_segment


_patch_exif()
_patch_mime_types()
