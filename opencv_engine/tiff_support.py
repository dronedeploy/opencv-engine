import logging
import uuid

import cv2
import gdal
import numpy
import osr


logger = logging.getLogger(__name__)

TIFF_FORMATS = {
    '.tiff': 'TIFF',
    '.tif': 'TIFF',
}


class TiffMixin(object):

    def read_tiff(self, buffer, create_alpha=True):
        """ Reads image using GDAL from a buffer, and returns a CV2 image.
        """

        offset = float(getattr(self.context, 'offset', 0.0)) or 0

        mem_map_name = '/vsimem/{}'.format(uuid.uuid4().hex)
        gdal_img = None
        try:
            gdal.FileFromMemBuffer(mem_map_name, buffer)
            gdal_img = gdal.Open(mem_map_name)

            channels = [gdal_img.GetRasterBand(i).ReadAsArray() for i in range(1, gdal_img.RasterCount + 1)]

            if len(channels) >= 3:  # opencv is bgr not rgb.
                red_channel = channels[0]
                channels[0] = channels[2]

                # Offset is z-offset to the elevation value
                # If it's set, we are reading a DEM tiff, which stores its elevation data in channels[2]
                # We don't want to add an offset to a no-data value
                no_data_value = None if not offset else gdal_img.GetRasterBand(1).GetNoDataValue()
                add_offset_if_data = numpy.vectorize(
                    lambda x: x + offset if offset and x != no_data_value else x, otypes=[numpy.float32])
                # If there's an offset, run add_offset_if_data on numpy array, else just assign it to the proper channel
                channels[2] = add_offset_if_data(red_channel) if offset else red_channel

            if len(channels) < 4 and create_alpha:
                self.no_data_value = gdal_img.GetRasterBand(1).GetNoDataValue()
                channels.append(numpy.float32(gdal_img.GetRasterBand(1).GetMaskBand().ReadAsArray()))

            return cv2.merge(channels)
        finally:
            gdal_img = None
            gdal.Unlink(mem_map_name)  # Cleanup.

    def read_vsimem(self, fn):
        """Read GDAL vsimem files"""
        vsifile = None
        try:
            vsifile = gdal.VSIFOpenL(fn, 'r')
            gdal.VSIFSeekL(vsifile, 0, 2)
            vsileng = gdal.VSIFTellL(vsifile)
            gdal.VSIFSeekL(vsifile, 0, 0)
            return gdal.VSIFReadL(1, vsileng, vsifile)
        finally:
            if vsifile:
                gdal.VSIFCloseL(vsifile)

    def write_channels_to_tiff_buffer(self, channels):
        """
        Writes tiff channels to buffer to be returned to user.

        IMPORTANT NOTE:
        This method will be called by the engine if one or both of the following filters are set:
         * format(tiff)
         * band_selector(n)

        If the band_selector filter has been used, a 32-bit tiff will be returned, and it will only include the
        specified band.

        Otherwise, an 8-bit tiff will be returned.

        Because of this logic, we are assuming that a user is requesting a DEM tiff if they use the
        band_selector filter to select a single tiff channel. In the future, we may want to change our code to require
        users to indicate whether they are requesting a DEM or an orthomosaic, since DEMs can include information on
        more than one channel. Currently, if a user requests a DEM and specifies the format(tiff) filter, they will
        receive an 8-bit DEM, which will have truncated values over 256 (which can be particularly problematic
        with elevations). The only internal code that is requesting DEMs is the exporter, but it won't run into the
        8-bit truncation issue because it is specifying format(tiff) and band_selector(0).

        Args:
            channels: tiff channels.

        Returns:
            gdal image buffer containing data in channels.

        """

        mem_map_name = '/vsimem/{}.tiff'.format(uuid.uuid4().hex)
        driver = gdal.GetDriverByName('GTiff')
        w, h = channels[0].shape
        gdal_img = None
        try:
            if len(channels) == 1:
                # DEM Tiff (32 bit floating point single channel)
                gdal_img = driver.Create(mem_map_name, w, h, len(channels), gdal.GDT_Float32)
                outband = gdal_img.GetRasterBand(1)
                outband.WriteArray(channels[0], 0, 0)
                outband.SetNoDataValue(-32767)
                outband.FlushCache()
                outband = None
                gdal_img.FlushCache()

                self.set_geo_info(gdal_img)
                return self.read_vsimem(mem_map_name)
            elif len(channels) == 4:
                # BGRA 8 bit unsigned int.
                gdal_img = driver.Create(mem_map_name, h, w, len(channels), gdal.GDT_Byte)
                band_order = [2, 1, 0, 3]
                img_bands = [gdal_img.GetRasterBand(i) for i in range(1, 5)]
                for outband, band_i in zip(img_bands, band_order):
                    outband.WriteArray(channels[band_i], 0, 0)
                    outband.SetNoDataValue(-32767)
                    outband.FlushCache()
                    del outband
                del img_bands

                self.set_geo_info(gdal_img)
                return self.read_vsimem(mem_map_name)
        finally:
            del gdal_img
            gdal.Unlink(mem_map_name)  # Cleanup.

    def set_geo_info(self, gdal_img):
        """ Set the georeferencing information for the given gdal image.
        """
        if hasattr(self.context, 'request') and hasattr(self.context.request, 'geo_info'):
            geo = self.context.request.geo_info
            gdal_img.SetGeoTransform([geo['upper_left_x'], geo['resx'], 0, geo['upper_left_y'], 0, -geo['resy']])

        # Set projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        gdal_img.SetProjection(srs.ExportToWkt())
        gdal_img.FlushCache()
        del srs
