import numpy
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
class tools_GIS(object):
    def __init__(self,folder_out=None):
        self.folder_out = folder_out
        return

#---------------------------------------------------------------------------------------------------------------------
    def get_bitmap_arcgis(self, lat=50, long=30, delta=10,W=800,H=800):
        array_lat = [lat - delta / 2, lat + delta / 2]
        array_long = [long - delta / 2, long + delta / 2]

        api = 'https://sampleserver3.arcgisonline.com/ArcGIS/rest/services/World/MODIS/ImageServer/exportImage'
        url = '{api}?f=image&bbox={coord}&size={W},{H}'.format(api=api, coord='%f,%f,%f,%f' % (array_long[0], array_lat[0], array_long[1], array_lat[1]),W=W,H=H)
        #url =  https://sampleserver3.arcgisonline.com/ArcGIS/rest/services/World/MODIS/ImageServer/exportImage?f=image&bbox=27,47,33,53&bboxSR=4300&size=800,800
        res = requests.get(url)
        image_bytes = io.BytesIO(res.content)
        img = Image.open(image_bytes)
        image = numpy.array(img)[:, :, [2, 1, 0]]
        return image

#---------------------------------------------------------------------------------------------------------------------
    def get_colors_cool(self,N):
        c1 = tools_draw_numpy.get_colors(256, colormap='Blues', alpha_blend=0.0, clr_blend=(255, 255, 255), shuffle=False)
        c2 = tools_draw_numpy.get_colors(256, colormap='Greens', alpha_blend=0.0, clr_blend=(255, 255, 255),shuffle=False)
        colors_cool = (6.0 * c1 + 4.0 * c2)[::-1] / 10
        res_colors = numpy.array([colors_cool[int(i)] for i in numpy.linspace(64, 192, N)])
        return res_colors
# ---------------------------------------------------------------------------------------------------------------------
    def get_colors_warm(self,N):

        colors_warm = tools_draw_numpy.get_colors(256, colormap='YlOrRd', alpha_blend=0.0, clr_blend=(0, 0, 0), shuffle=False)
        res_colors = numpy.array([colors_warm[int(i)] for i in numpy.linspace(64,192, N)])

        return res_colors
# ---------------------------------------------------------------------------------------------------------------------
    def fig_to_image(self,fig):
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', facecolor=(1, 1, 1))
        io_buf.seek(0)
        image = numpy.reshape(numpy.frombuffer(io_buf.getvalue(), dtype=numpy.uint8),newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:, :, [2, 1, 0]]
        return image
#----------------------------------------------------------------------------------------------------------------------
    def draw_points(self,df,idx_lat,idx_long,idx_value=None,idx_label=None,W=800,H=800,draw_terrain=False,value_by_size = False,edgecolor=(0.25, 0.25, 0.25, 1)):

        min_marker_size = 50
        max_marker_size = 800

        X = df.iloc[:, idx_long]
        Y = df.iloc[:, idx_lat]

        lat_range,long_range = [numpy.nanmin(X), numpy.nanmax(X)],[numpy.nanmin(Y), numpy.nanmax(Y)]
        pad = 0*numpy.abs(max((lat_range[1] - lat_range[0]) / 10, (long_range[1] - long_range[0]) / 10))

        lat_range[0]-=pad
        lat_range[1]+=pad
        long_range[0]-=pad
        long_range[1]+=pad

        fig = plt.figure(figsize=(W / 100, H / 100))
        stamen_terrain = cimgt.Stamen('terrain-background')
        projection = stamen_terrain.crs
        ax = plt.gca(projection=projection)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        #ax.add_feature(cfeature.LAND)

        ax.set_extent([lat_range[0], lat_range[1], long_range[0], long_range[1]], crs=ccrs.PlateCarree())

        if draw_terrain:
            ax.add_image(stamen_terrain, 8)

        if idx_value is not None:
            if value_by_size:
                S = df.iloc[:, idx_value]
                S-=numpy.nanmin(S)
                S=min_marker_size+(max_marker_size-min_marker_size)*S/(numpy.nanmax(S))
                # colors = (0,0.5,1,0)
                colors = (tools_draw_numpy.get_colors(X.shape[0], colormap='winter', shuffle=True))[:,[2, 1, 0]].astype(numpy.float32) / 255
            else:
                S = min_marker_size
                colors255 = (tools_draw_numpy.get_colors(256, colormap='jet', shuffle=False))[:,[2, 1, 0]].astype(numpy.float32) / 255
                V = df.iloc[:, idx_value].values
                V-=numpy.nanmin(V)
                V/=((numpy.nanmax(V))/255)
                V[numpy.isnan(V)]=0
                idx = [int(v) for v in V]
                colors = colors255[idx]
        else:
            S=min_marker_size
            colors = 'red'

        plt.scatter(x=X,y=Y,color=colors,s=S,alpha=1,transform=ccrs.PlateCarree(),edgecolor=edgecolor)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        image = self.fig_to_image(fig)

        return image
#---------------------------------------------------------------------------------------------------------------------
    def ex_kenia(self):


        fig = plt.figure()
        stamen_terrain = cimgt.Stamen('terrain-background')
        projection = stamen_terrain.crs
        ax = plt.gca(projection=projection)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        #ax.add_feature(cfeature.LAND)

        ax.set_extent([-23, 55, -35, 40])

        shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        kenya = [country for country in reader.records() if country.attributes["NAME_LONG"] == "Kenya"][0]

        shape_feature = ShapelyFeature([kenya.geometry], ccrs.PlateCarree(), facecolor="lime", edgecolor='black', lw=1)
        ax.add_feature(shape_feature)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        image = self.fig_to_image(fig)

        return image
#---------------------------------------------------------------------------------------------------------------------