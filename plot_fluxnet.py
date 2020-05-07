from ipywidgets import widgets, interact
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
from bokeh.palettes import RdBu11 as Mypalette
from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource, BoxZoomTool
output_notebook()
import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
#display(HTML(r"""<style id=hide>div.input{display:none;}</style><button type="button"onclick="var myStyle = document.getElementById('hide').sheet;myStyle.insertRule('div.input{display:inherit !important;}', 0);">Show notebook cells</button>"""))
# display(HTML(r"""<style id=hide>div.input{display:none;}</style>"""))


def esat(T):
    return 611.2*np.exp((17.62*T)/(-30.03+T+273.15))
    
def esat_slope(T):
    return esat(T)*(4284/(-30.03+273.15+T)**2)

varnames = {'timestamp':'date_time',
            'hour':'hour',
            'month':'month',
            'year':'year',
            'air temperature':'TA_F',
            'K_in':'SW_IN_F',
            'K_out': 'SW_OUT',
            'L_in':'LW_IN_F',
            'L_out':'LW_OUT',
#            'PAR':'PPFD_IN',
            'Q*': 'NETRAD',
            'VPD':'VPD_F',
            'RH':'RH',
            'preciptation':'P_F',
            'u*': 'USTAR',
            'wind speed': 'WS_F',
            'LE':'LE_CORR',
            'H': 'H_CORR',
            '[CO2]':'CO2_F_MDS',
            'NEE': 'NEE_VUT_MEAN',
            'respiration': 'RECO_NT_VUT_MEAN',
            'GPP': 'GPP_NT_VUT_MEAN',
            'albedo':'albedo',
            'ET_Makkink': 'ET_Makkink'}

units    =  {'timestamp':'date_time',
            'hour':'-',
            'month':'-',
            'year':'-',
            'air temperature':'C',
            'K_in':'W/m2',
            'K_out': 'W/m2',
            'L_in':'W/m2',
            'L_out':'W/m2',
#            'PAR':'PPFD_IN',
            'Q*': 'W/m2',
            'VPD':'hPa',
            'RH':'%',
            'preciptation':'mm',
            'u*': 'm/s',
            'wind speed': 'm/s',
            'LE':'W/m2',
            'H': 'W/m2',
            '[CO2]':'ppmV',
            'NEE':'umol/m2/s',
            'respiration':'umol/m2/s',
            'GPP':'umol/m2/s',
            'albedo': '-',
            'ET_Makkink': 'W/m2'}
numeric_name = 'TA_F'
aggr_methods = ['30min','day', 'month']

def myplot(site='Loobos',x_variable='K_in',y_variable='GPP',
           color_by='air temperature', averaging='day', plot_lines = False, 
           connect_points = False):
           # , range_min=0, range_max=1) :
		   # , range_min=0, range_max=1):
    fname=''
    range_min = 0
    range_max = 1
    if (averaging == 'day'):
        period = 'DD'
    elif( averaging == 'month'):
        period = 'MM'
    elif (averaging == '30min'):
        period = 'HH'
    if (site == 'Loobos'):
        fname_old = fname
        fname='FLX_NL-Loo_FLUXNET2015_FULLSET_%s_1996-2013_1-3.csv'%(period)
    elif (site == 'Horstermeer'):
        fname_old = fname
        fname='FLX_NL-Hor_FLUXNET2015_FULLSET_%s_2004-2011_1-3.csv'%(period)
    elif (site == 'Rollesbroich'):
        fname_old = fname
        fname='FLX_DE-RuR_FLUXNET2015_FULLSET_%s_2011-2014_1-3.csv'%(period)

    if (fname_old != fname):
        all_data=pd.read_csv(fname, na_values='-9999')
        if (averaging == 'day'):
            timestamp = all_data['TIMESTAMP']
            timestamp = pd.to_datetime(timestamp,format="%Y%m%d")
        elif (averaging == 'month'):
            timestamp = all_data['TIMESTAMP']
            timestamp = pd.to_datetime(timestamp,format="%Y%m")
        elif (averaging == '30min'):
            timestamp = all_data['TIMESTAMP_END']
            timestamp = pd.to_datetime(timestamp,format="%Y%m%d%H%M")
        all_data['TIMESTAMP'] = timestamp
        all_data = all_data.set_index('TIMESTAMP')
        all_data['date_time'] = all_data.index.values
        all_data['month'] = pd.DatetimeIndex(all_data['date_time']).month
        all_data['year'] = pd.DatetimeIndex(all_data['date_time']).year
        all_data['hour'] = pd.DatetimeIndex(all_data['date_time']).hour + pd.DatetimeIndex(all_data['date_time']).minute/60
        Temp = all_data[varnames['air temperature']]
        s = esat_slope(Temp)
        gamma = 65.5
        all_data['ET_Makkink'] = 0.65*(s/(gamma+s))*all_data[varnames['K_in']]
        all_data['albedo'] = all_data['SW_OUT'].values/all_data['SW_IN_F'].values
        all_data['albedo'] = np.where((all_data['albedo'] > 1), 1,all_data['albedo']) 
        all_data['albedo'] = np.where((all_data['albedo'] < 0), 0,all_data['albedo']) 
        es = esat(Temp)
        all_data['RH'] = 100*(1-all_data['VPD_F']/(0.01*es))

        loc_varnames = varnames.copy()
        for key in varnames.keys():
            if (varnames[key]  not in all_data):
                foo=loc_varnames.pop(key)

    my_x = all_data[varnames[x_variable]].values
    my_y = all_data[varnames[y_variable]].values
    my_c = all_data[varnames[color_by]].values

    # fig=plt.figure(figsize=(9,5))
    _tools_to_show = 'box_zoom,save,hover,reset' 
    if (x_variable =='timestamp'):
        p = figure(title=site, 
                   x_axis_label="%s (%s)"%(x_variable, units[x_variable]), 
                   y_axis_label="%s (%s)"%(y_variable, units[y_variable]),
                   tools=_tools_to_show, toolbar_location="below",
                   toolbar_sticky=False, x_axis_type="datetime", plot_height=500, plot_width=900)
        # p.toolbar.active_drag = BoxZoomTool()
    else:
        p = figure(title=site, 
                   x_axis_label="%s (%s)"%(x_variable, units[x_variable]), 
                   y_axis_label="%s (%s)"%(y_variable, units[y_variable]),
                   tools=_tools_to_show,toolbar_location="below",
                   toolbar_sticky=False, plot_height=500,  plot_width=900)
        # p.toolbar.active_drag = BoxZoomTool()
  
    data = {'x_values': my_x,
            'y_values': my_y,
            'c_values': my_c}

    source = ColumnDataSource(data=data)
    
    if (range_min >= range_max):
        range_min = range_max - 0.1
    if (color_by != 'timestamp'):
        mapper = LinearColorMapper( palette=Mypalette, 
                                   low=(np.nanmin(my_c)+range_min*(np.nanmax(my_c)-np.nanmin(my_c))), 
                                   high=(np.nanmin(my_c)+range_max*(np.nanmax(my_c)-np.nanmin(my_c))))
    else:
        mapper = LinearColorMapper( palette=Mypalette, 
                                   low=(np.min(my_c.astype(float))+range_min*(np.max(my_c.astype(float))-np.min(my_c.astype(float)))), 
                                   high=(np.min(my_c.astype(float))+range_max*(np.max(my_c.astype(float))-np.min(my_c.astype(float)))), )
    colors= { 'field': 'c_values', 'transform': mapper}
    
#    if (plot_line):
#         p.line(my_x, my_y, line_width=2)
#    else:
    p.scatter('x_values', 'y_values', source=source, fill_color=colors, line_color=None)
    if (connect_points):
        p.line(my_x,my_y)
    if (plot_lines and x_variable != 'timestamp' and color_by!= 'time_stamp'):
         if (x_variable =='timestamp'):
             my_x = my_x.astype(int)
         n_lines=4
         quant_step=1.0/(n_lines)
         q_low = np.arange(n_lines)*quant_step
         q_high = (np.arange(n_lines)+1)*quant_step
         n_points = 10
         x_step=1.0/(n_points)
         p_low = np.arange(n_points)*x_step
         p_high = (np.arange(n_points)+1)*x_step
         for i in range(n_lines):
             sum_x = np.zeros((n_points,), dtype=float)
             sum_y = np.zeros((n_points,), dtype=float)
             cond_c = (my_c > np.nanquantile(my_c, q_low[i]))* \
                      (my_c < np.nanquantile(my_c, q_high[i]))
             for j in range(n_points):
                cond_p = (my_x[cond_c] > np.nanquantile(my_x[cond_c], p_low[j]))* \
                         (my_x[cond_c] < np.nanquantile(my_x[cond_c], p_high[j]))
                sum_x[j] = np.nanmedian(my_x[cond_c][cond_p])
                sum_y[j] = np.nanmedian(my_y[cond_c][cond_p])
                
             palette_index = int(len(Mypalette)*(i/(n_lines)))
#             print (sum_x)
             p.line(sum_x, sum_y, line_width=5, line_color=Mypalette[palette_index])
 
    color_bar = ColorBar(color_mapper=mapper, label_standoff=4,  title=color_by, location=(0,0))
    p.add_layout(color_bar, 'right')
    show(p)

    
varname_keys=list(varnames.keys())
interact(myplot,site=['Loobos','Horstermeer','Rollesbroich'], x_variable=varnames.keys(), 
         y_variable=varnames.keys(), color_by=varnames.keys(),  averaging=aggr_methods,         
         plot_lines=False, connect_points = False);
#, range_min=(0,1,0.1),  range_max=(0,1,0.1));
