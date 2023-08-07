import json
import cv2
import os
import uuid
import numpy
import io
import folium
from folium import plugins
import inspect
from pyppeteer import launch
import asyncio
# ----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
class tools_GIS(object):
    def __init__(self,folder_out=None):
        self.folder_out = folder_out
        self.TP = tools_time_profiler.Time_Profiler()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_GPS_boundries(self,gis_points,W,H):
        filename_temp_html = uuid.uuid4().hex + '.html'
        filename_temp_png = filename_temp_html.split('.')[0] + '.png'
        self.build_folium_html(gis_points,filename_temp_html,mode='bbox')
        self.html_to_png(os.getcwd().replace('\\','/')+self.folder_out[1:]+filename_temp_html,filename_temp_png,W,H)

        image_bbox = cv2.imread(self.folder_out+filename_temp_png)
        os.remove(self.folder_out + filename_temp_html)
        os.remove(self.folder_out + filename_temp_png)


        V = tools_image.bgr2hsv(image_bbox)[:,:,2]
        xx = numpy.where(V>0)
        top = numpy.min(numpy.array(xx).T[:,0])
        left  = numpy.where(V[top] > 0)[0][0]
        right = numpy.where(V[top] > 0)[0][-1]
        bottom =V.shape[0]-top-10 +  numpy.sum(V[-top - 10:, left:right], axis=1).argmax()

        north = numpy.array(gis_points)[:, 0].max()
        south = numpy.array(gis_points)[:, 0].min()
        west = numpy.array(gis_points)[:, 1].min()
        east = numpy.array(gis_points)[:, 1].max()

        scaler = 1e5
        k_east_west = numpy.longdouble(scaler*(east - west) / (right - left))
        b_east_west = (east + west - k_east_west*(right+left)/scaler)/2
        bound_east,bound_west = b_east_west,b_east_west+k_east_west*image_bbox.shape[1]/scaler

        k_south_nord = numpy.longdouble(scaler * (south - north) / (bottom - top))
        b_south_nord = (south + north - k_south_nord * (bottom + top) / scaler) / 2
        bound_north, bound_south = b_south_nord, b_south_nord + k_south_nord * image_bbox.shape[0] / scaler

        #print(k_east_west,b_east_west,k_south_nord,b_south_nord)

        return bound_east,bound_west,bound_north, bound_south
# ----------------------------------------------------------------------------------------------------------------------
    def gps_to_ij(self, gis_points, W, H, dct_bbox):
        scaler = 1e5
        k_east_west = -numpy.longdouble(scaler * (dct_bbox['east'] - dct_bbox['west']) / (W))
        b_east_west = (dct_bbox['east'] + dct_bbox['west'] - k_east_west * (W) / scaler) / 2
        I = (gis_points[:, 1] - b_east_west) * scaler / k_east_west

        k_south_nord = -numpy.longdouble(scaler * (dct_bbox['north'] - dct_bbox['south']) / (H))
        b_south_nord = (dct_bbox['north'] + dct_bbox['south'] - k_south_nord * (H) / scaler) / 2
        J = (gis_points[:, 0] - b_south_nord) * scaler / k_south_nord
        #print(k_east_west, b_east_west, k_south_nord, b_south_nord)
        IJ = numpy.concatenate([[I],[J]]).astype(int).T
        return IJ
# ----------------------------------------------------------------------------------------------------------------------
    def build_folium_html(self, gis_points,filename_html,colors=None,mode=None):

        tiles = ['openstreetmap','stamentoner','mapquestopen','stamenterrain','cartodbpositron','cartodbdark_matter']
        map_folium = folium.Map(location=gis_points[0],tiles=tiles[1],zoom_control=True,scrollWheelZoom=True,dragging=True)

        if mode =='bbox':
            north = numpy.array(gis_points)[:, 0].max()
            south = numpy.array(gis_points)[:, 0].min()
            west = numpy.array(gis_points)[:, 1].min()
            east = numpy.array(gis_points)[:, 1].max()
            poly_bound = [(north, west), (north, east), (south, east), (south, west), (north, west)]
            folium.PolyLine(locations=poly_bound,color='#ff0000',weight=1).add_to(map_folium)

        weight,fillOpacity = 1,0.6

        if mode in ['clean','bbox']:
            weight, fillOpacity = 0, 0

        if colors is None:
            colors = ['#C00000']*gis_points.shape[0]

        for point,color in zip(gis_points,colors):
            folium.CircleMarker(location=point,radius=3,color=color,weight=weight,fillColor=color,fillOpacity=fillOpacity).add_to(map_folium)

        folium.CircleMarker(location=gis_points[ 0], radius=12, color=colors[ 0], weight=weight, fillColor=colors[ 0],fillOpacity=1).add_to(map_folium)
        folium.CircleMarker(location=gis_points[-1], radius=12, color=colors[-1], weight=weight, fillColor=colors[-1],fillOpacity=1).add_to(map_folium)

        map_folium.fit_bounds(map_folium.get_bounds())
        map_folium.save(self.folder_out + filename_html)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def build_folium_png_with_gps(self,gis_points, filename_out,W,H,draw_points=False):
        bound_east,bound_west,bound_north, bound_south = self.evaluate_GPS_boundries(gis_points,W,H)
        filename_temp_html = uuid.uuid4().hex + '.html'
        self.build_folium_html(gis_points, filename_temp_html,mode=(None if draw_points else 'clean'))
        self.html_to_png(os.getcwd().replace('\\', '/') + self.folder_out[1:] + filename_temp_html, filename_out,W,H)
        os.remove(self.folder_out + filename_temp_html)
        dct_bbox = dict({'east':bound_east,'west':bound_west,'north':bound_north, 'south':bound_south})
        json.dump(dct_bbox, open(self.folder_out+filename_out.split('.')[0]+'.json', 'w'))
        image = cv2.imread(self.folder_out +filename_out)
        return dct_bbox,image
# ----------------------------------------------------------------------------------------------------------------------
    async def __html_to_png_core(self,html_full_path, output_image_path,W,H):
        browser = await launch({'defaultViewport': {'width': W, 'height': H}})
        page = await browser.newPage()
        await page.goto('file://'+html_full_path)
        await page.screenshot({'path': output_image_path,'clip':{"x": 0,"y": 0,"width": W,"height": H}})
        await browser.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def html_to_png(self,html_full_path,output_image_path,W,H):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        asyncio.get_event_loop().run_until_complete(self.__html_to_png_core(html_full_path, self.folder_out+output_image_path,W,H))
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
