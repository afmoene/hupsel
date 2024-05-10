from ipywidgets import widgets, interact
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
# For pallettes, see: https://cjdoris.github.io/Bokeh.jl/stable/palettes/
from bokeh.palettes import Plasma11, Viridis11, RdBu11
from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource, BoxZoomTool, Band
from bokeh.resources import INLINE
#output_notebook(INLINE)
output_notebook()

import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML

# all colormaps
colormaps = {'plasma': Plasma11,
             'viridis': Viridis11,
             'rdbu': RdBu11}

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
            'T_a':'TA_F',
            'K_in':'SW_IN_F',
            'K_out': 'SW_OUT',
            'L_in':'LW_IN_F',
            'L_out':'LW_OUT',
            'PAR':'PPFD_IN',
            'Q*': 'NETRAD',
            'VPD':'VPD_F',
            'RH':'RH',
            'precipitation':'P_F',
            'u*': 'USTAR',
            'u_speed': 'WS_F',
            'LE':'LE_CORR',
            'H': 'H_CORR',
            '[CO2]':'CO2_F_MDS',
            'NEE': 'NEE_VUT_MEAN',
            'Reco': 'RECO_NT_VUT_MEAN',
            'GPP': 'GPP_NT_VUT_MEAN',
            'albedo':'albedo',
            'EF':'EF',
            'bowen': 'bowen',
            'DOY':'DOY',
            'LE_Makkink': 'LE_Makkink'}

units    =  {'timestamp':'date_time',
            'hour':'-',
            'month':'-',
            'year':'-',
            'T_a':'C',
            'K_in':'W/m2',
            'K_out': 'W/m2',
            'L_in':'W/m2',
            'L_out':'W/m2',
            'PAR':'µmolPhoton m-2 s-1',
            'Q*': 'W/m2',
            'VPD':'hPa',
            'RH':'%',
            'precipitation':'mm/day',
            'u*': 'm/s',
            'u_speed': 'm/s',
            'LE':'W/m2',
            'H': 'W/m2',
            '[CO2]':'ppmV',
            'NEE':'umol/m2/s',
            'Reco':'umol/m2/s',
            'GPP':'umol/m2/s',
            'albedo': '-',
            'EF': '-',
            'bowen': '-',
            'DOY': '-',
            'LE_Makkink': 'W/m2'}

# Make this variable so that students can list the variable names
var_names = varnames.keys()

# Folder for data
data_folder='data'

# Sites			
sites     =  ['Loobos','Horstermeer','Rollesbroich', 'Hyytiala', 'LeBray', 'WalnutCreek']
site_code =  ['NL-Loo','NL-Hor','DE-RuR', 'FI-Hyy', 'FR-LBr', 'US-Wkg']
site_start_y =  [1996,2004,2011,1996,1996,2004]
site_end_y   =  [2013,2011,2014,2014,2008, 2014]
site_range   =  ['1-3', '1-3', '1-3', '1-4', '1-4', '1-4']

# Averaging periods
aggr_methods = ['30min','day', 'month']

# Constuct units consistent with the averaging periods
aggr_units = [units.copy(), units.copy(), units.copy()]
aggr_units[1]['NEE'] = 'gC/m2/d'
aggr_units[2]['NEE'] = 'gC/m2/d'
aggr_units[1]['respiration'] = 'gC/m2/d'
aggr_units[2]['respiration'] = 'gC/m2/d'
aggr_units[1]['GPP'] = 'gC/m2/d'
aggr_units[2]['GPP'] = 'gC/m2/d'

# Availability of data per site
avail = {}
var_avail = {}
for varname in varnames.keys():
    avail[varname] = True
for i in range(len(sites)):
   var_avail[sites[i]] = avail.copy()
var_avail['Horstermeer']['PAR'] = False

def fluxplot(site='Loobos',x_var ='timestamp',y_var ='T_a',
           color_by=None, averaging='day', plot_lines = False, 
           n_lines=4, connect_points = False, plot_quant = False,
           colormap = "plasma",
           quantile = 0.25,
           return_data = False):

    """
	
    Flexible plot function to visualize FluxNet data 

    This function provides a flexible way to make scatter plots of FluxNet data. All arguments
    are optional (they all have a default value). Main uses of this function are:
    * plot a time series of a variable
    * plot a scatter plot linking two variables to each other
    * for scatter plots, the data can additionally be stratified (colored) by a third variable.
    * the relationship between the x-variable and y-variable can be clarified by showing a fit

    Parameters
    ----------
    site : string
        Name of the site from which to plot data (currently: "Loobos", "Horstermeer", "Rollesbroich")
    x_var : string
        Name of variable to plot on the x-axis
    y_var : string
        Name of variable to plot on the y-axis
    color_by : string
        Name of variable to plot on the y-axis
    averaging: string
        Averaging period of the displayed data (currenly: "month", "day", "30min")
    plot_lines: boolean (True/False)
        Plot a fit for the relation between the x-variable and y-variable; multiple lines are drawn,
        for multiple classes of the stratifying variable (the color_by variable)
    n_lines: integer
        If plot_lines: number of lines to fit (i.e. the number classes in which the stratifying variable is grouped)
    plot_quant: boolean (True/False)
        If plot_lines: show a band around the line of give quantiles (default: 0.25 and 0.75)
    quantile: float
        If plot_lines and plot_quant: the quantile to be shown (shows the range [quantile, 1-quantile])
    connect_points: boolean (True/False)
        Connect the plotted ppints (useful when plotting a timeseries)
    colormap: string ("plasma", "viridis", "rdbu")
        Name of colormap:
        * plasma (blue - purple - yellow)
        * viridis (purple - green - yellow)
        * rdbu (red - white - blue)
    return_data: boolean (True/False)
        Make the function return the dataset used for the plotting (all variables)
	
    Returns
    -------
    None
        By default, the function does not return a value (unless 'return_data' was specified: in that 
        case a Pandas data frame with the dataset is returned
   

    """
    if (not (site in sites)):
        print('%s is an unknown site. Choose from %s'%(site,sites))
        return
    if (not (x_var in varnames.keys())):
        print('%s is an unknown variable. Choose from %s'%(x_var,varnames.keys()))
        return
    if (not (y_var in varnames.keys())):
        print('%s is an unknown variable. Choose from %s'%(y_var,varnames.keys()))
        return
    if (color_by and (not (color_by in varnames.keys()))):
        print('%s is an unknown variable. Choose from %s'%(color_by,varnames.keys()))
        return
    if (not (averaging in aggr_methods)):
        print('%s is an unknown averaging interval. Choose from %s'%(averaging,aggr_methods))
        return
    if (not (colormap in colormaps.keys())):
        print('%s is an unknown colormap. Choose from %s'%(colormap,colormaps.keys()))
        return
    if (var_avail[site][x_var] == False):
        print('Variable %s is not available for site %s'%(x_var,site))
        return  
    if (var_avail[site][y_var] == False):
        print('Variable %s is not available for site %s'%(y_var,site))
        return
    if (color_by and (var_avail[site][color_by] == False)):
        print('Variable %s is not available for site %s'%(color_by,site))
        return
    if (plot_quant & (plot_lines == False)):
        print('Cannot show quantiles if plot_lines is not set to True')
        return

    # Set some initial values
    fname=''
    range_min = 0
    range_max = 1
    
    # Intialize pallette
    Mypalette = colormaps[colormap]
    
    # Determine components of file names
    if (averaging == 'day'):
        period = 'DD'
        local_units = aggr_units[1]
    elif( averaging == 'month'):
        period = 'MM'
        local_units = aggr_units[2]
    elif (averaging == '30min'):
        period = 'HH'
        local_units = aggr_units[0]
    site_num = sites.index(site)
    fname_old = fname
    fname=data_folder+'/'+'FLX_%s_FLUXNET2015_FULLSET_%s_%i-%i_%s.csv'%(site_code[site_num], period, site_start_y[site_num],site_end_y[site_num],site_range[site_num],)

    # Did we change file name -> read the new file and compute derived variables
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
        Temp = all_data[varnames['T_a']]
        s = esat_slope(Temp)
        gamma = 65.5
        all_data['LE_Makkink'] = 0.65*(s/(gamma+s))*all_data[varnames['K_in']]
        all_data['albedo'] = all_data[varnames['K_out']].values/all_data[varnames['K_in']].values
        all_data['albedo'] = np.where((all_data['albedo'] > 1), 1,all_data['albedo']) 
        all_data['albedo'] = np.where((all_data['albedo'] < 0), 0,all_data['albedo']) 
        all_data['bowen'] = all_data[varnames['H']].values/all_data[varnames['LE']].values
        all_data['EF'] = all_data[varnames['LE']].values/(all_data[varnames['LE']].values + all_data[varnames['H']].values)
        es = esat(Temp)
        all_data['RH'] = 100*(1-all_data['VPD_F']/(0.01*es))
        all_data['DOY'] = pd.DatetimeIndex(all_data['date_time']).dayofyear

        loc_varnames = varnames.copy()
        for key in varnames.keys():
            if (varnames[key]  not in all_data):
                foo=loc_varnames.pop(key)

    # Get the variables to be plotted
    my_x = all_data[varnames[x_var]].values
    my_y = all_data[varnames[y_var]].values
    if (color_by):
        my_c = all_data[varnames[color_by]].values
    if (color_by):  
        data = {'x_values': my_x,
                'y_values': my_y,
                'c_values': my_c}
    else:
        data = {'x_values': my_x,
                'y_values': my_y}  
        
        
    # fig=plt.figure(figsize=(9,5))
    _tools_to_show = 'box_zoom,save,pan,reset' 

    if (x_var =='timestamp'):
        p = figure(title="%s (averaging: %s)"%(site, averaging), 
                   x_axis_label="%s (%s)"%(x_var, local_units[x_var]), 
                   y_axis_label="%s (%s)"%(y_var, local_units[y_var]),
                   tools=_tools_to_show, toolbar_location="below",
                   toolbar_sticky=False, x_axis_type="datetime", height=500, width=900)
        # p.toolbar.active_drag = BoxZoomTool()
    else:
        p = figure(title="%s (averaging: %s)"%(site, averaging), 
                   x_axis_label="%s (%s)"%(x_var, local_units[x_var]), 
                   y_axis_label="%s (%s)"%(y_var, local_units[y_var]),
                   tools=_tools_to_show,toolbar_location="below",
                   toolbar_sticky=False, height=500,  width=900)
        # p.toolbar.active_drag = BoxZoomTool()
    
    source = ColumnDataSource(data=data)
    
    if (range_min >= range_max):
        range_min = range_max - 0.1
    if (color_by):
        if (color_by != 'timestamp'):
            mapper = LinearColorMapper( palette=Mypalette, 
                                       low=(np.nanmin(my_c)+range_min*(np.nanmax(my_c)-np.nanmin(my_c))), 
                                       high=(np.nanmin(my_c)+range_max*(np.nanmax(my_c)-np.nanmin(my_c))))
        else:
            mapper = LinearColorMapper( palette=Mypalette, 
                                       low=(np.min(my_c.astype(float))+range_min*(np.max(my_c.astype(float))-np.min(my_c.astype(float)))), 
                                       high=(np.min(my_c.astype(float))+range_max*(np.max(my_c.astype(float))-np.min(my_c.astype(float)))), )
        colors= { 'field': 'c_values', 'transform': mapper}
    else:
        colors = 'black'
    
#    if (plot_line):
#         p.line(my_x, my_y, line_width=2)
#    else:
    p.scatter('x_values', 'y_values', source=source, fill_color=colors, line_color=None)
    if (connect_points):
        p.line(my_x,my_y)
    if (color_by and plot_lines and x_var != 'timestamp' and color_by!= 'time_stamp'):
         if (x_var =='timestamp'):
             my_x = my_x.astype(int)
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
             upper = np.zeros((n_points,), dtype=float)       
             lower = np.zeros((n_points,), dtype=float)
             cond_c = (my_c > np.nanquantile(my_c, q_low[i]))* \
                      (my_c < np.nanquantile(my_c, q_high[i]))
             for j in range(n_points):
                cond_p = (my_x[cond_c] > np.nanquantile(my_x[cond_c], p_low[j]))* \
                         (my_x[cond_c] < np.nanquantile(my_x[cond_c], p_high[j]))
                sum_x[j] = np.nanmedian(my_x[cond_c][cond_p])
                sum_y[j] = np.nanmedian(my_y[cond_c][cond_p])
                if (plot_quant):
                    upper[j] = np.nanquantile(my_y[cond_c][cond_p],1 - quantile)
                    lower[j] = np.nanquantile(my_y[cond_c][cond_p],quantile)

             if (plot_quant):
                 df = pd.DataFrame(data=dict(x=sum_x, lower=lower, upper=upper)).sort_values(by="x")
                 source = ColumnDataSource(df.reset_index())
             palette_index = int(len(Mypalette)*((i+0.5)/(n_lines))) # Add 0.5 to ensure that the color is in the middle of the range of colored data points
             p.line(sum_x, sum_y, line_width=5, line_color=Mypalette[palette_index])
             if (plot_quant):
                 band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay', 
                             fill_alpha=0.7, line_width=1, line_color='black', fill_color=Mypalette[palette_index])
                 p.add_layout(band)
    if (color_by):
        color_bar = ColorBar(color_mapper=mapper, label_standoff=4,  title="%s (%s)"%(color_by, local_units[color_by]), location=(0,0))
        p.add_layout(color_bar, 'right')
    p.toolbar.active_drag = None
    show(p)
	
    if (return_data):
        return all_data

    
#varname_keys=list(varnames.keys())
#interact(myplot,site=['Loobos','Horstermeer','Rollesbroich'], x_var=varnames.keys(), 
#         y_var=varnames.keys(), color_by=varnames.keys(),  averaging=aggr_methods,         
#         plot_lines=False, connect_points = False);
##, range_min=(0,1,0.1),  range_max=(0,1,0.1));
