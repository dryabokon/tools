import numpy
import pandas as pd
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
# ---------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_DF

# ---------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'


# ----------------------------------------------------------------------------------------------------------------------
class Plotly_builder:
    def __init__(self, dark_mode=False):
        self.dark_mode = dark_mode
        self.turn_light_mode()
        self.id_text_box = 'ID00'
        self.id_plot_01 = 'ID01'
        self.id_plot_02 = 'ID02'
        self.app = dash.Dash(external_stylesheets=([dbc.themes.BOOTSTRAP]))
        return

    # ---------------------------------------------------------------------------------------------------------------------
    def turn_light_mode(self, dark_mode=None):
        if dark_mode is None:
            dark_mode = self.dark_mode

        if dark_mode:
            self.clr_bg = '#2b2b2b'
            self.clr_font = '#FFFFFF'
        else:
            self.clr_bg = "#EEEEEE"
            self.clr_font = '#000000'
        return

    # ---------------------------------------------------------------------------------------------------------------------
    def build_layout(self):

        figure = self.create_figure_test_sales()
        layout = html.Div([
            html.H4(id=self.id_text_box),
            html.Div([dcc.Graph(id=self.id_plot_01, figure=figure)]),
        ])

        return layout

    # ---------------------------------------------------------------------------------------------------------------------
    def run_server(self, port=8050):

        self.app.layout = self.build_layout()
        self.set_callback()
        self.app.run_server(port=port, debug=True, use_reloader=False)

        return

    # ----------------------------------------------------------------------------------------------------------------------
    def create_figure_test_cofee(self):
        df = pd.read_csv(
            'https://raw.githubusercontent.com/plotly/datasets/96c0bd/sunburst-coffee-flavors-complete.csv')
        fig = go.Figure(go.Treemap(ids=df.ids, labels=df.labels, parents=df.parents, level='ids'))
        fig.update_layout(
            uniformtext=dict(minsize=10, mode='hide'),
            margin=dict(t=50, l=25, r=25, b=25))
        return fig

    # ---------------------------------------------------------------------------------------------------------------------
    def create_figure_test_sales(self):

        A = numpy.array(
            [['East', 'Houston', 'AX', 42, 9], ['North', 'Hartley', 'BI', 24, 8], ['West', 'Brewster', 'BS', 33, 14],
             ['West', 'Presidio', 'DQ', 40, 15], ['South', 'Duval', 'DS', 42, 9], ['North', 'Motley', 'EH', 44, 34],
             ['North', 'Gray', 'ES', 49, 31], ['South', 'Duval', 'FJ', 20, 7], ['South', 'Zavala', 'FL', 31, 24],
             ['East', 'Tyler', 'GT', 30, 4], ['South', 'Duval', 'GW', 20, 6], ['East', 'Rusk', 'HW', 20, 8],
             ['North', 'Dallam', 'IJ', 20, 6], ['West', 'Brewster', 'IK', 33, 20], ['South', 'Zavala', 'IV', 28, 20],
             ['East', 'Tyler', 'JC', 21, 17], ['North', 'Dallam', 'JE', 35, 23], ['West', 'Brewster', 'JF', 20, 8],
             ['South', 'Webb', 'KR', 20, 6], ['South', 'Hidalgo', 'LQ', 41, 16], ['South', 'Hidalgo', 'LR', 39, 7],
             ['South', 'Hidalgo', 'LV', 21, 11], ['North', 'Motley', 'MW', 22, 21], ['North', 'Motley', 'NY', 35, 15],
             ['East', 'Shelby', 'NZ', 45, 16], ['North', 'Floyd', 'OH', 31, 6], ['West', 'Presidio', 'OT', 35, 9],
             ['East', 'Rusk', 'OY', 40, 20], ['North', 'Hartley', 'PL', 42, 37], ['South', 'Webb', 'QJ', 46, 40],
             ['East', 'Houston', 'QK', 22, 8], ['West', 'Pecos', 'QM', 48, 45], ['South', 'Webb', 'RR', 48, 26],
             ['West', 'Presidio', 'SV', 45, 30], ['North', 'Moore', 'TW', 33, 6], ['North', 'Moore', 'TZ', 33, 20],
             ['North', 'Floyd', 'UA', 44, 34], ['East', 'Tyler', 'UF', 49, 35], ['East', 'Rusk', 'UM', 24, 8],
             ['East', 'Shelby', 'VN', 29, 6], ['North', 'Hartley', 'WE', 39, 37], ['North', 'Floyd', 'WH', 44, 37],
             ['West', 'Pecos', 'WV', 30, 25], ['West', 'Pecos', 'WZ', 37, 23], ['East', 'Shelby', 'XH', 47, 6],
             ['North', 'Gray', 'XY', 31, 28], ['South', 'Zavala', 'YA', 25, 21], ['North', 'Gray', 'YJ', 35, 13],
             ['East', 'Houston', 'YN', 38, 23], ['North', 'Moore,', 'ZM', 21, 13], ['North', 'Dallam', 'ZQ', 49, 13]])
        df = pd.DataFrame(A, columns=['region', 'county', 'salesperson', 'calls', 'sales'])
        df[['calls', 'sales']] = df[['calls', 'sales']].astype(int)
        dct_function = {'cols_metric': ['sales', 'calls'], 'metric_function': (lambda x, y: x / y)}
        df_hierarchical = tools_DF.build_hierarchical_dataframe(df,
                                                                cols_labels_level=['salesperson', 'county', 'region'],
                                                                col_size='calls', dct_function=dct_function)

        dct_marker_colors = dict(colors=df_hierarchical['metric'], colorscale='RdBu')
        fig = go.Sunburst(labels=df_hierarchical['id'], parents=df_hierarchical['parent_id'],
                          values=df_hierarchical['size'],
                          branchvalues='total',
                          marker=dct_marker_colors,
                          hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Success rate: %{color:.2f}')
        fig = go.Figure(fig)

        self.df_sales_hierarchical = df_hierarchical
        return fig

    # ---------------------------------------------------------------------------------------------------------------------
    def create_figure_sunbust(self, df_hierarchical, colorscale='viridis', metric_name='', height=600,
                              do_treemap=False):

        # colors = tools_draw_numpy.values_to_colors(df_hierarchical['metric'], '~RdBu')
        # dct_marker_colors = dict(colors=[tools_draw_numpy.BGR_to_HTML(c[[2,1,0]]) for c in colors])
        dct_marker_colors = dict(colors=df_hierarchical['metric'], colorscale=colorscale,
                                 line=dict(color='#222222', width=1))
        hovertemplate = None
        if metric_name != '':
            hovertemplate = '%{label}<br>size: %{value}<br>'
            hovertemplate += metric_name + ':%{color:.2f}'

        if do_treemap:
            fig = go.Treemap(labels=df_hierarchical['id'], parents=df_hierarchical['parent_id'],
                             values=df_hierarchical['size'], branchvalues='total', marker=dct_marker_colors,
                             hovertemplate=hovertemplate, hoverinfo='none')
        else:
            fig = go.Sunburst(labels=df_hierarchical['id'], parents=df_hierarchical['parent_id'],
                              values=df_hierarchical['size'], branchvalues='total', marker=dct_marker_colors,
                              hovertemplate=hovertemplate, hoverinfo='none')

        fig = go.Figure(fig, layout=go.Layout(font={'size': 20}))
        fig.update_layout(autosize=True, width=500, margin=dict(t=0, l=0, r=0, b=0))
        fig.layout.plot_bgcolor = '#FFFFFF'
        fig.layout.paper_bgcolor = '#FFFFFF'
        return fig

    # ----------------------------------------------------------------------------------------------------------------------
    def create_figure_bar(self, labels, values=None, tickvals_x=None, categoryorder=True, font_size=12,
                          orientation='h'):

        df = pd.DataFrame({'x': 1 if values is None else [v for v in values], 'label': [str(l) for l in labels]})
        fig = go.Figure(go.Bar(x=df['x'], y=df['label'], orientation=orientation, hoverinfo='none'),
                        layout=go.Layout(bargap=0.02, font={'size': font_size}))
        fig.update_layout(autosize=True, height=font_size * 1.5 * df.shape[0], margin=dict(t=0, l=0, r=0, b=0))

        fig.update_layout(xaxis={'visible': True, 'showticklabels': False, 'zeroline': False})
        fig.update_layout(yaxis={'showgrid': False, 'title': None, 'showticklabels': True,
                                 'ticklabelposition': 'outside', 'color': self.clr_font, 'zeroline': False})

        if categoryorder:
            # values shold bot be 0 !!
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        else:
            fig.update_yaxes(autorange="reversed")

        fig.update_xaxes(tickvals=tickvals_x, showgrid=True if tickvals_x is not None else False, gridwidth=1,
                         gridcolor=self.clr_font)

        fig.update_traces(marker={'line': dict(color=self.clr_bg, width=0)})
        fig.layout.plot_bgcolor = self.clr_bg
        fig.layout.paper_bgcolor = self.clr_bg

        return fig

    # ----------------------------------------------------------------------------------------------------------------------
    def create_figure_scatter(self, X, Y, labels=None, color=None, idx_marker=None, color_marker=None):
        df = pd.DataFrame({'x': [x for x in X], 'y': [y for y in Y]})

        if idx_marker is not None and color_marker is not None:
            color = [color] * df.shape[0] + [color_marker]
            df = df.append({'x': X[idx_marker], 'y': Y[idx_marker]}, ignore_index=True)

        marker = dict(size=16, color=color, showscale=False)

        fig = go.Figure(data=go.Scatter(x=df['x'], y=df['y'], mode='markers', text=[l for l in labels], marker=marker))
        fig.update_layout(autosize=True, margin=dict(t=0, l=0, r=0, b=0))
        fig.update_layout(xaxis={'visible': False, 'showticklabels': False, 'zeroline': False})
        fig.update_layout(yaxis={'visible': False, 'showticklabels': False, 'zeroline': False})
        fig.update_xaxes(showgrid=False)

        # fig.update_traces(marker={'line':dict(color=self.clr_bg,width=0)})
        fig.layout.plot_bgcolor = self.clr_bg
        fig.layout.paper_bgcolor = self.clr_bg

        return fig

    # ----------------------------------------------------------------------------------------------------------------------
    def create_figure_squarify(self, labels, sizes, colors=None, W=200, H=200):
        import squarify
        sizes = numpy.array(sizes)
        sss = squarify.normalize_sizes(sizes / sum(sizes), W, H)
        sq_res = numpy.array([(el['x'], el['y'], el['dx'], el['dy']) for el in squarify.squarify(sss, 0, 0, W, H)])
        padding = 1
        sq_res[:, 0] += padding
        sq_res[:, 2] -= 2 * padding
        sq_res[:, 1] += padding
        sq_res[:, 3] -= 2 * padding

        rect_x, rect_y, text_x, text_y = [], [], [], []
        for sq, label in zip(sq_res, labels):
            left, top, right, bottom = sq[0], sq[1], sq[0] + sq[2], sq[1] + sq[3]
            rect_x.append([left, right, right, left, None])
            rect_y.append([top, top, bottom, bottom, None])
            text_x.append((left + right) / 2)
            text_y.append((top + bottom) / 2)

        rect_x = numpy.array(rect_x).flatten()
        rect_y = numpy.array(rect_y).flatten()

        if colors is None:
            dct_marker_colors = dict(color='rgba(238,238,238,0)', opacity=0)
        else:
            dct_marker_colors = dict(color=[tools_draw_numpy.BGR_to_HTML(c[[2, 1, 0]]) for c in colors])

        hoverinfo = 'none'
        fig = go.Figure(
            go.Scatter(x=rect_x, y=rect_y, fill="toself", hoverinfo=hoverinfo, mode='lines', marker=dct_marker_colors,
                       opacity=1.0, showlegend=False))
        fig.add_trace(go.Scatter(x=text_x, y=text_y, text=labels, mode="text", hoverinfo=hoverinfo, showlegend=False))

        # fig = go.Figure()
        # fig.add_shape(type="rect",xref="x", yref="y",x0=2, y0=0, x1=3, y1=2,line=dict(color="RoyalBlue",width=3,),fillcolor="LightSkyBlue",text='A')
        # fig.add_shape(type="rect",xref="x", yref="y",x0=3, y0=1, x1=4, y1=4,line=dict(color="RoyalBlue", width=3, ), fillcolor="LightSkyBlue",text='B')

        fig.update_layout(xaxis={'visible': False, 'showticklabels': False})
        fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
        fig.layout.plot_bgcolor = '#FFFFFF'
        fig.layout.paper_bgcolor = '#FFFFFF'

        return fig

    # ----------------------------------------------------------------------------------------------------------------------
    def set_callback(self):
        # ----------------------------------------------------------------------------------------------------------------------
        @self.app.callback(
            Output(self.id_text_box, "children"),
            Input(self.id_plot_01, "clickData"),
        )
        def callbal_function(clickData):
            path_to_label, label = '', ''
            if clickData:
                clickData = clickData['points'][0]
                if 'label' in clickData: label = clickData['label']
                if 'currentPath' in clickData: path_to_label = clickData['currentPath']
                print(path_to_label + label)
            return path_to_label + label
# ---------------------------------------------------------------------------------------------------------------------
