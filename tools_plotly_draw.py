import io
from io import BytesIO
import pandas as pd

from dash import dcc,html
import dash_bootstrap_components as dbc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import base64
# ----------------------------------------------------------------------------------------------------------------------
class DashDraw(object):

    #dct_style_image = {'textAlign': 'center', 'height': '150px', 'margin': '0px', 'padding': '0px', 'object-fit': 'scale-down'}
    # dct_style_image = {'margin': '0px', 'padding': '0px','object-fit': 'scale-fit'}
    # dct_style_text  = {'margin': '0px', 'padding': '0px','textAlign': 'center','object-fit': 'scale-fit'}
    style_scroll = {'height':'50%', 'overflow-y':'scroll'}

    # style_fit = {'margin': '0px', 'padding': '0px',
    #              'object-fit': 'scale-fit',
    #              'transition': 'transform 3s ease',
    #              'transform': 'scale(1)'}




# ----------------------------------------------------------------------------------------------------------------------
    def __init__(self,dark_mode=False):

        self.dark_mode = dark_mode

        if dark_mode:
            self.clr_bg = "#2B2B2B"
            self.clr_fg = "#FFFFFF"
            self.clr_pad = "#222222"
            self.clr_grid = "#404040"
            self.clr_banner = "#404040"
            self.clr_header = "2B2B2B"
            self.clr_sub_header = "#214646"
            self.plotly_template =  'plotly_dark'
        else:
            self.clr_bg = "#FFFFFF"
            self.clr_fg = "#000000"
            self.clr_pad = "#EEEEEE"
            self.clr_grid = "#C0C0C0"
            self.clr_banner = "#E0E0E0"
            self.clr_header = "#2B2B2B"
            self.clr_sub_header = "#83B4B4"
            self.plotly_template = 'plotly_white'

        return
# ----------------------------------------------------------------------------------------------------------------------
    def filename_img_to_uri(self, filename):
        buffer_img = open(filename, 'rb').read()
        encoded = base64.b64encode(buffer_img)
        result = 'data:image/png;base64,{}'.format(encoded)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def fig_to_uri(self, in_fig, close_all=True, **save_args):
        out_img = BytesIO()
        in_fig.savefig(out_img, format='png', **save_args, facecolor=in_fig.get_facecolor())
        if close_all:
            in_fig.clf()
            plt.close('all')
        out_img.seek(0)  # rewind file
        encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)
# ----------------------------------------------------------------------------------------------------------------------
    def draw_item(self,children=None,id=None,framed=False,style=None):

        layout = html.Div(children=children, id=id, style=None)
        if framed:
            layout = dbc.Card(dbc.CardBody(layout,style=style),color=style['background-color'] if (style is not None and 'background-color' in style) else None)

        return layout
# ----------------------------------------------------------------------------------------------------------------------
    def draw_card(self,children=None,id=None,color="#404040",style=None):
        asset = html.H5(id=id,children=children)
        layout = dbc.Card(dbc.CardBody(asset,style=style),color=color,inverse=True)
        return layout
# ----------------------------------------------------------------------------------------------------------------------
    def draw_graph(self, id, figure,framed=False,style=None,remove_bar=True):
        graph = dcc.Graph(id=id, figure=figure, config={"displayModeBar": False} if remove_bar else None)

        if framed:
            layout = dbc.Card(dbc.CardBody(graph,style=style,className="no-scrollbars"),color=style['background-color'] if (style is not None and 'background-color' in style) else None)
        else:
            layout = html.Div(graph,style=style,className="no-scrollbars")

        return layout
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
    def parse_data(self,contents):
        content_type, content_string = contents.split(',')
        df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')),delimiter='\t')
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def prepare_icon(self):
        # from PIL import Image
        # img = Image.open('./assets/Image1.png')
        # img.save(self.folder_out+'favicon.ico', format='ICO', sizes=[(32, 32)])
        return
# ----------------------------------------------------------------------------------------------------------------------