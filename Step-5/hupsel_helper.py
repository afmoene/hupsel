import pandas as pd 
import numpy as np
import datetime
from numpy import exp
from bokeh.plotting import figure, output_file, show, output_notebook, ColumnDataSource
from bokeh.palettes import Category10
from bokeh.palettes import RdBu11 as Mypalette
from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource, BoxZoomTool, Band, Range1d

import itertools
import os # For reading data file

# Define reasonable ranges for meteo data
good_range_T = [273-40,273+50]   # K
good_range_q = [1e-4,1e-1]       # kg/kg
good_range_p = [900e2,1100e2]    # Pa
good_range_Kin = [0,1365]        # W/m2
good_range_Qnet = [-200,1200]    # W/m2
good_range_G = [-300,500]        # W/m2
good_range_ra = [1e-3,1000]      # s/m
good_range_rc = [1e-3,1000]      # s/m
good_range_u = [1e-3,50]         # m/s -> to prevent division by zero in ra; the 
good_range_zu = [0.5,300]        # m
good_range_zT = [0.5,300]        # m
good_range_d = [0,0.45]          # m -> to prevent negative (z-d)
good_range_z0 = [1e-5,10]        # m
good_range_z0h = [1e-6,1]        # m

var_ranges_dict = {
  "K_in" : good_range_Kin,
  "T"   : good_range_T,
  "q"   : good_range_q,
  "p"   : good_range_p,
  "Q_net" : good_range_Qnet,
  "G"   : good_range_G,
  "ra"  : good_range_ra,    
  "rc"  : good_range_rc,
  "u"   : good_range_u,    
  "zu"  : good_range_zu,    
  "zT"  : good_range_zT,    
  "d"   : good_range_d,    
  "z0"  : good_range_z0,    
  "z0h" : good_range_z0h,    
}

def my_warning(text):
    from colorama import Fore, Back, Style
    print(Fore.MAGENTA + "Warning: " + text)
    print(Style.RESET_ALL)
    
def my_error(text):
    from colorama import Fore, Back, Style
    print(Fore.RED + "Error: " + text)
    print(Style.RESET_ALL)

# Function to cycle plotting colors (see https://stackoverflow.com/questions/39839409/when-plotting-with-bokeh-how-do-you-automatically-cycle-through-a-color-pallett) 
def color_gen():
    yield from itertools.cycle(Category10[10])

def checkplot(x, f_ref, f_in, x_name, f_name):
    output_notebook()
    p = figure(x_axis_label="%s"%(x_name), 
               y_axis_label="%s"%(f_name))
    p.line(x,f_ref(x), legend_label='correct %s'%(f_name))
    p.scatter(x,f_in(x), legend_label='your %s'%(f_name))
    show(p)
    
def check_function(f_in, f_ref, f_name, var_name):
    var=[]
    nargs = len(var_name)
    for vname in var_name:
        var_range = np.linspace( var_ranges_dict[vname][0], var_ranges_dict[vname][1] )
        var.append(var_range)
        
    error=[]
    for i in range(nargs):
        var_in = []
        for j in range(nargs):
            if (j==i):
                var_in.append(var[j])
            else:
                var_in.append(var[j].mean())
        if (nargs == 1):
            ref_data = f_ref(var_in[0])
            test_data = f_in(var_in[0])
        elif (nargs == 2):
            ref_data = f_ref(var_in[0], var_in[1])
            test_data = f_in(var_in[0], var_in[1])
        elif (nargs == 3):
            ref_data = f_ref(var_in[0], var_in[1], var_in[2])
            test_data = f_in(var_in[0], var_in[1], var_in[2])
        elif (nargs == 4):
            ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3])
            test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3])
        elif (nargs == 5):
            ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4])
            test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4])
        elif (nargs == 6):
            ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5])
            test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5])
        elif (nargs == 7):
            ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5], var_in[6])
            test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5], var_in[6])
        else:
            my_error("check_function: cannot deal with functions with %i arguments"%(nargs))
            
        rms = np.sqrt(np.mean((ref_data - test_data)**2))
        if (rms < 1e-3*abs(ref_data.mean())):
            error.append(0)
        else:
            error.append(1)
                
    if (sum(error) == 0):
        print("Well done")
    else:
        print("Not good")
        for i in range(len(var)):
            var_in = []
            for j in range(len(var)):
                if (j==i):
                    var_in.append(var[j])
                else:
                    var_in.append(var[j].mean())
            if (nargs == 1):
                ref_data = f_ref(var_in[0])
                test_data = f_in(var_in[0])
            elif (nargs == 2):
                ref_data = f_ref(var_in[0], var_in[1])
                test_data = f_in(var_in[0], var_in[1])
            elif (nargs == 3):
                ref_data = f_ref(var_in[0], var_in[1], var_in[2])
                test_data = f_in(var_in[0], var_in[1], var_in[2])
            elif (nargs == 4):
                ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3])
                test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3])
            elif (nargs == 5):
                ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4])
                test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4])
            elif (nargs == 6):
                ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5])
                test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5])
            elif (nargs == 7):
                ref_data = f_ref(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5], var_in[6])
                test_data = f_in(var_in[0], var_in[1], var_in[2], var_in[3], var_in[4], var_in[5], var_in[6])
            else:
                my_error("check_function: cannot deal with functions with %i arguments"%(nargs))

            output_notebook()
            x_name = var_name[i]
            # f_name is now passed as an argument to this function
            f_name = f_name
            pl = figure(x_axis_label="%s"%(x_name), 
                        y_axis_label="%s"%(f_name))
            pl.line(var[i],ref_data, legend_label='correct %s'%(f_name))
            pl.scatter(var[i],test_data, legend_label='your %s'%(f_name))
            show(pl)

def f_days_since_rain(precipitation, threshold = 0.0):
    """
    Determine number of days since last rain, based on daily precipitation data. 
    Input:
        precipitation : time series of daily rainfall amount (presumably mm/day)
        threshold     : the amount of daily rainfall that needs to be exceeded to classify
                        a day as a day with rain
    Return:
        days_since_rain: an array with the number of days since last rain (0 means: it rained that day

    """
    days_since_rain = np.where(precipitation > threshold, 0,1)
    for i in range(1,len(precipitation)):
        if (days_since_rain[i]):
            days_since_rain[i] += days_since_rain[i-1]
    return days_since_rain

def myplot(*args, **kwargs):
    """

    Flexible plot function to visualize your data 

    There are two major ways to invoke the function: with and without a Pandas dataframe:
    
    myplot(df, ['varname1', 'varname2'] )
        This will use data from the dataframe df to plot. The variable 'varname1' will be used
        as the x-variable, variable 'varname2' will be used as the y-variable.
    myplot( [x, y] )
        The will use the data contained in variables x and y (arrays) to plot
    
    To plot more than one series in a plot, you can repeat the x/y combination in square brackets:
    myplot(df, ['varname1', 'varname2'], ['varname1', 'varname3'] )
    myplot( [x, y], [x,y2] )
     
    * plot one series as a line: `myplot(df,['Date','K_in'])`. 
      The x-name and y-name are given as a list (enclosed in square brackets).
    * plot two series as lines: `myplot(df,['Date','K_in'], ['Date','K_out_m'])`. 
      The second series is also given as a list, separated from the first list with comma.

    Additional options for series
    ----------
    * plot a series with something other than a line 
       * `myplot(df,['Date','K_in','-'])`: plot a line
       * `myplot(df,['Date','K_in','o'])`: plot dots
       * `myplot(df,['Date','prec','#'])`: bar graph (only one series per graph)
    * give each series a name, to be used in the legend (you then also must give the plot-type)
       * myplot(df, ['Date', 'K_in','-', 'Global radiation'], ['Date', 'K_out', 'o', 'Reflected shortwave radiation'] )
       * myplot([time, albedo, '-', 'Surface albedo'])

    Additional keyword arguments
    ----------
    xlabel, ylabel:       label on x-axis and/or y-axis 
                          e.g. myplot( [x, y], xlabel='time (hour)', ylabel='temperature (K)')
    color_by:             color dots based on the value of a third variable; when plotting from a dataframe, give
                          the name of the variable, otherwise give the variable itself
                          e.g. myplot(df, ['Date', 'LvE_m', 'o'], color_by = 'T_1_5')
                          or   myplot([x, y, 'o'], color_by = c) 
    x_axis_type, y_axis_type:
                          'linear' or 'log' axis
                          e.g. myplot([x, y, 'o'], x_axis_type = 'linear', y_axis_type = 'log') 
    xlim, ylim:           limits for x-axis or y-axis
                          e.g. myplot([x, y, 'o'], xlim = [0,10])
    
    Returns
    -------
    None
  

    """
    # Process args
    if (type(args[0]) == pd.DataFrame):
        df = args[0]
        series_list = args[1:]
        df_plot = True
    else:
        # Copy list of series (assuming those are the only arguments)
        series_list = args
        df_plot = False
    
    # Process kwargs
    my_xlabel = None
    my_ylabel = None
    do_color_by = False
    xtype = 'linear'
    ytype = 'linear'
    xlim = None
    ylim = None
    for key, value in kwargs.items():
        if (key == 'xlabel'):
            my_xlabel = str(value)
        elif (key == 'ylabel'):
            my_ylabel = str(value)
        elif (key == 'color_by'):
            if (df_plot):
                do_color_by = True
                color_by = value
            else:
                do_color_by = True
                color_by = value
        elif (key == 'x_axis_type'):
            if (value == "log"):
                xtype = "log"
            elif (value == "linear"):
                xtype = "linear"
            else:
                my_warning("myplot: unknown value %s for keyword argument:%s"%(value, key))    
        elif (key == 'y_axis_type'):
            if (value == "log"):
                ytype = "log"
            elif (value == "linear"):
                ytype = "linear"
            else:
                my_warning("myplot: unknown value %s for keyword argument:%s"%(value, key))           
        elif (key == 'xlim'):
            if (not type(value) in (list, tuple)): 
                my_error("myplot: value for keyword xlim should be a list or a tuple")
            else:
                xlim = tuple(value)
        elif (key == 'ylim'):
            if (not type(value) in (list, tuple)): 
                my_error("myplot: value for keyword ylim should be a list or a tuple")
            else:
                ylim = tuple(value)
        else:
            my_warning("myplot: unknown keyword argument:%s"%(key))
    
    # Set default scatter size
    scatter_size = 7
    
    # Check if more than one bar graph is asked for
    nbar = 0
    for serie in series_list:
        if (len(serie)>2):
            if (serie[2] == '#'):
                nbar += 1
    if (nbar > 1):
        print("You ask for more than one bar graph. We cannot handle that yet.")
        return
    
  
    if (do_color_by):
        # Needed?
        range_min = 0
        range_max = 1
        if (range_min >= range_max):
            range_min = range_max - 0.1 
        if (df_plot):
            my_c = df[color_by].values
        else:
            my_c = color_by
        mapper = LinearColorMapper( palette=Mypalette, 
                                    low=(np.nanmin(my_c)+range_min*(np.nanmax(my_c)-np.nanmin(my_c))), 
                                    high=(np.nanmin(my_c)+range_max*(np.nanmax(my_c)-np.nanmin(my_c))))
        colors = { 'field': 'c_values', 'transform': mapper}


        
    if (df_plot):
        # Check if variables are available in dataframe
        for s in series_list:
            for i in range(2):
                if (s[i] not in df.columns):
                    print("Variable %s does not exist in Dataframe"%(s[i]))
                    return

        # Check if units is present as attribute of dataframe
        if ('units' not in df.attrs.keys()):
            units = pd.DataFrame(len(df.keys())*[' '], columns=[df.keys()])
        else:
            units = df.attrs['units']

        # Start plot
        if (type(df[series_list[0][0]].values[0]) == np.datetime64):
            xtype = 'datetime'
        output_notebook()
        if (not my_xlabel):
            my_xlabel = "%s (%s)"%(series_list[0][0], units[series_list[0][0]])
        if (not my_ylabel):
            my_ylabel = "%s (%s)"%(series_list[0][1], units[series_list[0][1]])
       

        p = figure(width=800, height=400, 
                   x_axis_type=xtype, y_axis_type=ytype,
                   x_axis_label=my_xlabel, 
                   y_axis_label=my_ylabel)
        if (xlim):
            p.x_range=Range1d(xlim[0], xlim[1])
        if (ylim):
            p.y_range=Range1d(ylim[0], ylim[1])
        
        # Start color iterator
        color = color_gen()
        
        # Add a line for each series
        for s in series_list:
            # Plot type
            plottype='line'
            if (len(s)>2):
                if (s[2] == '-'):
                    plottype = 'line'
                elif (s[2] == 'o'):
                    plottype = 'scatter'
                elif (s[2] == '#'):
                    plottype = 'bar'
                else:
                    print("Unkown plot type: '%s'"%(s[2]))
                    return
                
            # Label for legend    
            if (len(s)>3):
                series_label = str(s[3])
            else:
                series_label = s[1]
            # do plot
            if (plottype == 'line'):
                p.line(df[s[0]],df[s[1]], legend_label=series_label, color=next(color))
            elif (plottype == 'scatter'):
                mycolor = color
                if (do_color_by):
                    my_x = df[s[0]].values
                    my_y = df[s[1]].values
                    # my_c was defined before)
                    data = {'x_values': my_x, 'y_values': my_y, 'c_values': my_c}
                    source = ColumnDataSource(data=data)
                    p.scatter('x_values', 'y_values', source=source, legend_label=series_label, \
                              fill_color=colors, line_color=None, size=scatter_size)
                    color_bar = ColorBar(color_mapper=mapper, label_standoff=4, location=(0,0))
                    p.add_layout(color_bar, 'right')
                else:
                    p.scatter(df[s[0]],df[s[1]], legend_label=series_label, fill_color=next(color), size=scatter_size)
            elif (plottype == 'bar'):
                barwidth = df[s[0]][1]-df[s[0]][0]
                p.vbar(x=df[s[0]], top=df[s[1]], width = 0.3*barwidth, \
                       legend_label=series_label, color=next(color))

        show(p)
    else:

        # We check that the lists contain data series
        for i in range(len(series_list)):
            s = series_list[i]
            # check that x and y are arrays
            if (not ((type(s[0]) == np.ndarray) or (type(s[0]) == pd.pandas.core.series.Series)) ):
                my_error("first variable in plot series # %i is not an array"%(i+1))
                return
            if (not ((type(s[1]) == np.ndarray) or (type(s[1]) == pd.pandas.core.series.Series)) ):
                my_error("second variable in plot series # %i is not an array"%(i+1))
                return
      
        # Check x-variable of first series: if it is time, we have a special x-axis
        if (type(series_list[0][0].values[0]) == np.datetime64):
            xtype = 'datetime'

        output_notebook()
        
        # Fix problem with autoscaling of log axis (Bokeh version May 2021)
        if (ytype == 'log'):
            s = series_list[0]
            cond = np.isfinite(s[1])
            s[0] = s[0][cond]
            s[1] = s[1][cond]
            if (do_color_by):
                my_c = my_c[cond]
        if (xtype == 'log'):
            s = series_list[0]
            cond = np.isfinite(s[0])
            s[0] = s[0][cond]
            s[1] = s[1][cond]
            if (do_color_by):
                my_c = my_c[cond]
				
		# Fix problem with change in keywords: plot_width -> width and plot_height -> height (Bokeh version 3.1.1 installed on Cocalc somewhere in July 2023)
        p = figure(width=800, height=400, x_axis_type=xtype, y_axis_type=ytype,
                   x_axis_label=my_xlabel, 
                   y_axis_label=my_ylabel)
        if (xlim):
            p.x_range=Range1d(xlim[0], xlim[1])
        if (ylim):
            p.y_range=Range1d(ylim[0], ylim[1])
            
        # Start color iterator
        color = color_gen()
        # add a line for each series
        for s in series_list:
            # Check that series are of equal length
            if (len(s[0]) != len(s[1])):
                print("Series are not of equal length: %i and %i"%(len(s[0]), len(s[1])))
                      
            # Plot type and color
            plottype='line'
            if (len(s)>2):
                if (s[2] == '-'):
                    plottype = 'line'
                elif (s[2] == 'o'):
                    plottype = 'scatter'
                elif (s[2] == '#'):
                    plottype = 'bar'
                else:
                    print("Unkown plot type: '%s'"%(s[2]))
                    return
            # Label for legend    
            if (len(s)>3):
                series_label = str(s[3])
            else:
                series_label = ''

            mycolor = color
            
            # do plot
            if (plottype == 'line'):
                p.line(s[0],s[1], legend_label=series_label, color=next(color))
            elif (plottype == 'scatter'):
                mycolor = color
                if (do_color_by):
                    my_x = s[0][:]
                    my_y = s[1][:]
                    # my_c was defined before)
                    data = {'x_values': my_x, 'y_values': my_y, 'c_values': my_c}
                    source = ColumnDataSource(data=data)
                    p.scatter('x_values', 'y_values', source=source, legend_label=series_label, \
                              fill_color=colors, line_color=None, size=scatter_size)
                    color_bar = ColorBar(color_mapper=mapper, label_standoff=4, location=(0,0))
                    p.add_layout(color_bar, 'right') 
                else:
                    p.scatter(s[0],s[1], legend_label=series_label, fill_color=next(color), size=scatter_size)
            elif (plottype == 'bar'):
                barwidth = s[0][1]-s[0][0]
                p.vbar(x=s[0].values, top=s[1].values, legend_label=series_label, width=0.9*barwidth, color=next(color))

        # show the results
        show(p)
    
def myreadfile(fname, type='day'):   
    #teacher_dir = os.getenv('TEACHER_DIR')
    # fullpath = os.path.join(teacher_dir, 'JHL_data', fname)
    fullpath = fname
    
    if (type == 'day'):
        sheet_name = 0
    elif (type == '30min'):
        sheet_name = '30min Data'
    else:
        my_error('myreadfile: unknown data type %s'%(type))
        
    # The dataframe that contains the data (both KNMI data and MAQ data)
    if (type == 'day'):
        df = pd.read_excel(fullpath,skiprows=[0,1,2,3,5,6], sheet_name=sheet_name, parse_dates=[1])
    if (type == '30min'):
        df = pd.read_excel(fullpath,skiprows=[0,1,2,3,5,6], sheet_name=sheet_name, parse_dates=[0])
        df['Year'] = df['Date'].dt.year
        # The three lines below I added to work with the data from 2011; dont know if they are general.
        # I also changed the parse_dates above to [0] rather than [1]
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day        
        df = df.rename(columns={'HH': 'Hour', 'MM': 'Minute'})
        df['Date_start'] = pd.to_datetime(df[['Year','Month', 'Day','Hour', 'Minute']],
                                          format="%Y%m%d%H%M")-datetime.timedelta(seconds=30*60)
        df['Date_end'] = pd.to_datetime(df[['Year','Month', 'Day','Hour', 'Minute']],
                                          format="%Y%m%d%H%M")
        df['Date'] = df['Date_start'] + datetime.timedelta(seconds=15*60)
        df['Time'] = df['Date'].dt.hour + df['Date'].dt.minute / 60.0
        # This is not general, will not work always (used for 2014 data).
        df['TER'] = 2.5e-7 + 0*df['FCO2_m']
        df['GPP'] = - df['FCO2_m'] + df['TER']
      
    # Add the units (read from row 5) as an attribute to the dataframe
    units = pd.read_excel(fullpath,skiprows=[0,1,2,3], sheet_name=sheet_name, nrows=1) 
    units_dict = {}
    for i in range(len(units.values[0])):
        units_dict[df.keys()[i]] = units.values[0][i]
        df.attrs['units']=units_dict
    # Add variables that we just constructed
    df.attrs['units']['TER'] = df.attrs['units']['FCO2_m']
    df.attrs['units']['GPP'] = df.attrs['units']['FCO2_m']
        
    # Add description of variables
    descr = pd.read_excel(fullpath,skiprows=[0,1,2,3,5], sheet_name=sheet_name, nrows=1) 
    descr_dict = {}
    for i in range(len(descr.values[0])):
        descr_dict[df.keys()[i]] = descr.values[0][i]
        df.attrs['description']=descr_dict    
    # Add variables that we just constructed
    df.attrs['description']['TER'] = 'Estimate of terrestrial respiration'
    df.attrs['description']['GPP'] = 'Estimate of gross primary production (approx. photosynthesis), taking positive for CO2 uptake'

    # Assume that we want to use the first column (datetime) as an index
    df.set_index(df.keys()[0], inplace=True, drop=False)        

    return df

# Function to compute latent heat of vapourization
# The argument T is assumed to be in Kelvin
# This function is complete and functioning as an example
def f_Lv_ref(T):
    # Define constants
    c1 = 2501000
    c2 = 0.00095
    c3 = 273.15
    
    # Compute the result
    result =  c1*(1 - c2*(T - c3))
    
    return result   

def f_Lv(T):
    """
    Compute latent of vapourization of water, as a function of temperature
    Input:
        T             : temperature (Kelvin)
    Return:
        latent heat of vapourization of water (J/kg)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_T[0] <= np.array(T)) & (np.array(T) <= good_range_T[1] )).all()):
        my_warning("f_Lv: are you sure that the units of your temperature data are correct?")
    return f_Lv_ref(T)
    
# Function to compute saturated vapour pressure in Pa
# The argument T is assumed to be in Kelvin
# See secton 7.1 of the AVSi formularium
def f_esat_ref(T):
    # Define constants (check the values, the zeros are certainly wrong)
    c1 = 611.2
    c2 = 17.62
    c3 = 273.15
    c4 = 30.03
    
    # Compute the result (the structure of the equation is correct)
    result = c1*np.exp((c2*(T-c3))/(-c4+T))
    
    return result
def f_esat(T):
    """
    Compute saturated water vapour pressure, as a function of temperature
    Input:
        T             : temperature (Kelvin)
    Return:
        saturated water vapour pressure (Pa)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_T[0] <= np.array(T)) & (np.array(T) <= good_range_T[1] )).all()):
        my_warning("f_esat: are you sure that the units of your temperature data are correct?")
    return f_esat_ref(T)

# Function to compute slope of the saturated vapour pressure in Pa/K
# The argument T is assumed to be in Kelvin
# See secton 7.1 of the AVSi formularium
def f_s_ref(T):
    # Define constants (check the values, the zeros are certainly wrong)
    c1 = 4284
    c2 = 30.03

    # Compute the result (complete the formula)
    result = f_esat_ref(T)*c1/(-c2+T)**2
    
    return result

def f_s(T):
    """
    Compute the slope of the saturated water vapour pressure, as a function of temperature
    Input:
        T             : temperature (Kelvin)
    Return:
        slope of the saturated water vapour pressure (Pa/K)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_T[0] <= np.array(T)) & (np.array(T) <= good_range_T[1] )).all()):
        my_warning("f_s: are you sure that the units of your temperature data are correct?")
    return f_s_ref(T)

# Function to compute the psychrometer constant
# The arguments are temperature T (in K), pressure p in Pa, specific humidity q in kg/kg
def f_gamma_ref(T, p, q):
    # Define constants (chaeck the values, the zeros are certainly wrong)
    c1 = 65.5
    c2 = 0.84
    c3 = 0.00095
    c4 = 273.15
    c5 = 101300.0

    # Compute the result (complete the formula)
    result = c1*((1+c2*q)/(1-c3*(T-c4)))*(p/c5)
    
    return result  

def f_gamma(T, p, q):
    """
    Compute the psychrometer constant
    Input:
        T             : temperature (Kelvin)
        p             : pressure (Pa)
        q             : specific humidity (kg/kg)
    Return:
        psychrometer constant  (Pa/K)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_T[0] <= np.array(T)) & (np.array(T) <= good_range_T[1] )).all()):
        my_warning("f_gamma: are you sure that the units of your temperature data are correct?")
    if (not ((good_range_p[0] <= np.array(p)) & (np.array(p) <= good_range_p[1] )).all()):
        my_warning("f_gamma: are you sure that the units of your pressure data are correct?")
    if (not ((good_range_q[0] <= np.array(q)) & (np.array(q) <= good_range_q[1] )).all()):
        my_warning("f_gamma: are you sure that the units of your specific humidity data are correct?")
    return f_gamma_ref(T, p, q)

# Function to compute the specific heat of humid air
# The argument is specific humidity q in kg/kg
def f_cp_ref(q):
    # Define constants (chaeck the values, the zeros are certainly wrong)
    c1 = 0.84
    
    result = 1004*(1+c1*q)
    
    return result

def f_cp(q):
    """
    Compute the specific heat of air at constant pressure
    Input:
        q             : specific humidity (kg/kg)
    Return:
        specific heat at constant pressure (J/kg/K)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_q[0] <= np.array(q)) & (np.array(q) <= good_range_q[1] )).all()):
        my_warning("f_cp: are you sure that the units of your specific humidity data are correct?")
    return f_cp_ref(q) 
    
# Function to compute reference evapotranspiration according to Makkink
# The arguments are global radiation K_in in W/m2, temperature T (in K), pressure p in Pa, specific humidity q in kg/kg
# See secton 7.7 of the AVSI formularium or chapter 7 in Moene & van Dam (2014)
# Please note what is the unit of the resulting number !
def f_makkink_ref(K_in, T, p, q):
    # First compute s and gamma from the data
    gamma_data = f_gamma_ref(T, p, q)
    s_data = f_s_ref(T)
    
    # Now construct the Makkink equation (i.e. replace the '0' by the correct equation)
    # What is the unit?
    result  = 0.65*(s_data/(s_data + gamma_data))*K_in
    
    return result

def f_makkink(K_in, T, p, q):
    """
    Compute the reference evapotranspiration akkording to the Makkink method
    Input:
        K_in          : global radiation (W/m2)
        T             : temperature (K)
        p             : pressure (Pa)
        q             : specific humidity (kg/kg)
    Return:
        reference evapotranspidation according to Makkink (W/m2)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_Kin[0] <= np.array(K_in)) & (np.array(K_in) <= good_range_Kin[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your global radiation data are correct?")
    if (not ((good_range_T[0] <= np.array(T)) & (np.array(T) <= good_range_T[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your temperature data are correct?")
    if (not ((good_range_p[0] <= np.array(p)) & (np.array(p) <= good_range_p[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your pressure data are correct?")
    if (not ((good_range_q[0] <= np.array(q)) & (np.array(q) <= good_range_q[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your specific humidity data are correct?")
    return f_makkink_ref(K_in, T, p, q)

# Function to compute reference evapotranspiration according to Priestly-Taylor
# Input
#    Q_net : net radiation (W/m2)
#    G     : soil heat flux (W/m2)
#    T     : temperature (Kelvin)
#    p     : pressure (Pa)
#    q     : specific humidity (kg/kg)
# Output
#    LvEref: reference evapotranspiration according to Priestley-Taylor (W/m2)
#
# See secton 7.7 of the AVSI formularium, chapter 7 in Moene & van Dam (2014), 
# or the supporting document linked to at the intro of this exercise.
# Please note what is the unit of the resulting number !
def f_PT_ref(Q_net, G, T, p, q):
    # First compute s and gamma from the data
    gamma = f_gamma_ref(T, p, q)
    s = f_s_ref(T)
    
    # Now construct the Priestley-Taylor equation (i.e. replace the '0' by the correct equation)
    # What is the unit?
    result  = 1.26 * (s / (s + gamma)) * (Q_net - G)
    
    return result

def f_PT(Q_net, G, T, p, q):
    """
    Ccompute reference evapotranspiration according to Priestly-Taylor
    Input:
       Q_net : net radiation (W/m2)
       G     : soil heat flux (W/m2)
       T     : temperature (Kelvin)
       p     : pressure (Pa)
       q     : specific humidity (kg/kg)
    Return:
      reference evapotranspiration according to Priestley-Taylor (W/m2)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_Qnet[0] <= np.array(Q_net)) & (np.array(Q_net) <= good_range_Qnet[1] )).all()):
        my_warning("f_PT: are you sure that the units of your net radiation data are correct?")
    if (not ((good_range_G[0] <= np.array(G)) & (np.array(G) <= good_range_G[1] )).all()):
        my_warning("f_PT: are you sure that the units of your soil heat flux data are correct?")
    if (not ((good_range_T[0] <= np.array(T)) & (np.array(T) <= good_range_T[1] )).all()):
        my_warning("f_PT: are you sure that the units of your temperature data are correct?")
    if (not ((good_range_p[0] <= np.array(p)) & (np.array(p) <= good_range_p[1] )).all()):
        my_warning("f_PT: are you sure that the units of your pressure data are correct?")
    if (not ((good_range_q[0] <= np.array(q)) & (np.array(q) <= good_range_q[1] )).all()):
        my_warning("f_PT: are you sure that the units of your specific humidity data are correct?")    
    return f_PT_ref(Q_net, G, T, p, q)


# Function to compute aerodynamic resistance for neutral conditions (consistent with the 
# FAO method)
# Input
#    u     : wind speed (m/s)
#    zu    : height at which wind speed was measured (m)
#    zT    : height at which temperature and humidity was measured (m)
#    d     : displacement height (m)
#    z0    : roughness length for momentum
#    z0h   : roughness length for heat
# Output
#    ra    : aerodynamic resistance (s/m)
#
# See he supporting document linked to at the intro of this exercise.
# See secton 7.3 of the AVSI formularium, chapter 7 in Moene & van Dam (2014), 
def f_ra_ref(u, zu, zT, d, z0, z0h):
    # Note: you will need the natural logarithm (which we write in math as ln(x)).
    # In Python this is a function from the numpy library (imported as np: it is called log. Hence
    # the natural log of x would be computed as: answer = np.log(x)
    karman = 0.4
    
    # Compute the aerodynamic resistance (i.e. replace the zero by an apprpriate expression)    
    result = np.log((zu-d)/z0)*np.log((zT-d)/z0h)/(karman**2 * u)
    
    return result

def f_ra(u, zu, zT, d, z0, z0h):
    """
    Compute aerodyamic resistance for neautral conditions 
    Input:
       u     : mean horizontal wind speed (m/s)
       zu    : observation height of wind speed (m)
       zT    : observation height of temperature (m)
       d     : displacement height (m)
       z0    : roughness length for momentum (m)
       z0h   : roughness length for heat (m)
    Return:
      aerodynamic resistance for neutral conditions (s/m)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_u[0] <= abs(np.array(u))) & (abs(np.array(u)) <= good_range_u[1] )).all()):
        my_warning("f_ra: are you sure that the units of your wind speed data are correct?")
    if (not ((good_range_zu[0] <= np.array(zu)) & (np.array(zu) <= good_range_zu[1] )).all()):
        my_warning("f_ra: are you sure that the units of wind speed observation height are correct?")
    if (not ((good_range_zT[0] <= np.array(zT)) & (np.array(zT) <= good_range_zT[1] )).all()):
        my_warning("f_ra: are you sure that the units of temperature  observation height are correct?")
    if (not ((good_range_d[0] <= np.array(d)) & (np.array(d) <= good_range_d[1] )).all()):
        my_warning("f_ra: are you sure that the units of displacement height are correct?")
    if (not ((good_range_z0[0] <= np.array(z0)) & (np.array(z0) <= good_range_z0[1] )).all()):
        my_warning("f_ra: are you sure that the units of the momentum roughness length are correct?")
    if (not ((good_range_z0h[0] <= np.array(z0h)) & (np.array(z0h) <= good_range_z0h[1] )).all()):
        my_warning("f_ra: are you sure that the units of the heat roughness length are correct?")
    if (not ( np.array(z0h) < np.array(z0) ).all()):
        my_warniung("f_ra: roughness length for heat is usually smaller than roughness length for momentum")

    return f_ra_ref(u, zu, zT, d, z0, z0h)

# Function to compute reference evapotranspiration according to Penman_Monteith
# Input
#    Q_net : net radiation (W/m2)
#    G     : soil heat flux (W/m2)
#    T     : temperature (Kelvin)
#    p     : pressure (Pa)
#    q     : specific humidity (kg/kg)
#    ra    : aerodynamic resistance (s/m)
#    rc    : canopy resistance (s/m)
# Output
#    LvEref: reference evapotranspiration according to Penman_Monteith (W/m2)
#
# See secton 7.7 of the AVSI formularium, chapter 7 in Moene & van Dam (2014), 
# or the supporting document linked to at the intro of this exercise.
# Please note what is the unit of the resulting number !
def f_PM_ref(Q_net, G, T, p, q, ra, rc):
    # First compute s, gamma and cp from the data 
    # (those functions have already been defined, f_cp is new, but we made it for you)
    gamma = f_gamma(T, p, q)
    s = f_s(T)
    cp = f_cp(q)
    
    # In Penman-Monteith we use vapour pressure (e_a) as the variable for water vapour
    # content. We already have specific humidity (q) available within the function, 
    # so e_a can be determined to a reasonable approximation from q = (Rd/Rv) * (e_a/p)
    # (check the formularium how to do this)
    Rd = 287.0
    Rv = 462.0
    e_a = (Rv/Rd)* q * p
    
    # In Penman-Monteith we need the air density (rho_a). Using the gas law we can determine
    # rho_a using pressure, temperature: p = rho_a * R * T (where R is the gas constant for humid
    # air)
    # again: see the formularium)
    R = 287.0*(1+0.61*q)
    rho_a = p / (R * T)
       
    # Now construct the Penman-Monteith equation (i.e. replace the '0' by the correct equation)
    # It can be helpful to split the equation in a number of chunks (e.g. compute the denominator 
    # first) combine those chunks at the end
    # What is the unit?
    denom = s + gamma*(1+rc/ra)
    numer1 = s*(Q_net - G)
    numer2 = (rho_a*cp/ra)*(f_esat(T) - e_a)
    result  = (numer1 + numer2)/denom
    
    return result

def f_PM(Q_net, G, T, p, q, ra, rc):
    """
    Compute reference evapotranspiration according to Penman_Monteith
    Input:
       Q_net : net radiation (W/m2)
       G     : soil heat flux (W/m2)
       T     : temperature (Kelvin)
       p     : pressure (Pa)
       q     : specific humidity (kg/kg)
       ra    : aerodynamic resistance (s/m)
       rc    : canopy resistance (s/m)
    Return:
       reference evapotranspiration according to Penman_Monteith (W/m2)
    """
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_Qnet[0] <= np.array(Q_net)) & (np.array(Q_net) <= good_range_Qnet[1] )).all()):
        my_warning("f_PM: are you sure that the units of your net radiation data are correct?")
    if (not ((good_range_G[0] <= np.array(G)) & (np.array(G) <= good_range_G[1] )).all()):
        my_warning("f_PM: are you sure that the units of your soil heat flux data are correct?")
    if (not ((good_range_T[0] <= np.array(T)) & (np.array(T) <= good_range_T[1] )).all()):
        my_warning("f_PM: are you sure that the units of your temperature data are correct?")
    if (not ((good_range_p[0] <= np.array(p)) & (np.array(p) <= good_range_p[1] )).all()):
        my_warning("f_PM: are you sure that the units of your pressure data are correct?")
    if (not ((good_range_q[0] <= np.array(q)) & (np.array(q) <= good_range_q[1] )).all()):
        my_warning("f_PM: are you sure that the units of your specific humidity data are correct?")    
    if (not ((good_range_ra[0] <= np.array(ra)) & (np.array(ra) <= good_range_ra[1] )).all()):
        my_warning("f_PM: are you sure that the units of your aerodynamic resistance data are correct?")    
    if (not ((good_range_rc[0] <= np.array(rc)) & (np.array(rc) <= good_range_rc[1] )).all()):
        my_warning("f_PM: are you sure that the units of your canopy resistance data are correct?")    
    return f_PM_ref(Q_net, G, T, p, q, ra, rc)

def check_ra(f_ra_in):
    check_function(f_ra_in, f_ra_ref, 'ra', ['u', 'zu', 'zT', 'd', 'z0', 'z0h'])

def check_Lv(f_Lv_in):
    check_function(f_Lv_in, f_Lv_ref, 'Lv', ['T'])
        
def check_esat(f_esat_in):
    check_function(f_esat_in, f_esat_ref, 'esat', ['T'])
        
def check_s(f_s_in):
    check_function(f_s_in, f_s_ref, 's', ['T'])
                
def check_gamma(f_gamma_in):
    check_function(f_gamma_in, f_gamma_ref, 'gamma', ['T', 'p', 'q'])
              
def check_makkink(f_makkink_in):
    check_function(f_makkink_in, f_makkink_ref, 'makkink', ['K_in','T', 'p', 'q'])
    
def check_PT(f_PT_in):
    check_function(f_PT_in, f_PT_ref, 'Priestley-Taylor', ['Q_net', "G", 'T', 'p', 'q'])

def check_PM(f_PM_in):
    check_function(f_PM_in, f_PM_ref, 'Penman-Monteith', ['Q_net', "G", 'T', 'p', 'q', 'ra', 'rc'])

def check_crop_factor(cf):
    warning = 0
    if (np.sum(cf < 0)):
        my_warning("crop factors should be positive")
        warning += 1
    if (np.sum(cf > 2)):
        my_warning("crop factors larger dan 2 are quite unlikely")
        warning += 1
    if ( (type(cf[0]) == np.int64) | (type(cf[0]) == np.int32) ):
        my_warning("your crop factor should be a real number, not an integer")
        warning += 1
    if (np.isnan(cf).any()):
        my_warning("your crop factor contains a not-a-number")
        warning += 1
    if (warning == 0):
        print("Your values seem to be reasonable (no obious erros in terms of incorrect number type or extreme values")
        my_warning("This does not mean that they are correct.")
      
        
def check_ET(ET_in):
    warning = 0
    if (np.sum(ET_in < 0)):
        my_warning("actual evapotranspiration is usually positive")
        warning += 1
    if (np.sum(ET_in > 10)):
        my_warning("actual evapotranspiration above 10 mm/day is quite unlikely")
        warning += 1
    if (np.mean(np.abs(ET_in)) < 1e-2):
        my_warning("your actual evapotranspiration seems quite low, check your calculation and units")
        warning += 1
    if ( (type(ET_in[0]) == np.int64) | (type(ET_in[0]) == np.int32) ):
        my_warning("your actual evapotranspiration should be a real number, not an integer")
        warning += 1
    if (np.isnan(ET_in).any()):
        my_warning("your actual evapotranspiration contains a not-a-number")
        warning += 1
    if (warning == 0):
        print("Your values seem to be reasonable (no obious erros in terms of incorrect number type or extreme values")
        my_warning("This does not mean that they are correct.")
        
def f_declination(Gamma):
    c0 =  0.006918
    c1 = -0.399912
    s1 =  0.070257
    c2 = -0.006758
    s2 =  0.000907
    c3 = -0.002697
    s3 =  0.00148
    result = c0 + c1*np.cos(Gamma) + s1*np.sin(Gamma) + \
                  c2*np.cos(2*Gamma) + s2*np.sin(2*Gamma) + \
                  c3*np.cos(3*Gamma) + s3*np.sin(3*Gamma)
                   
    return result

def f_equation_of_time(Gamma):
    a0 =  3.8197
    c0 =  0.000075
    c1 =  0.001868
    s1 = -0.032077
    c2 = -0.014615
    s2 = -0.04089

    result = a0 * (c0 + c1*np.cos(Gamma) + s1*np.sin(Gamma) + c2*np.cos(Gamma) + s2*np.cos(Gamma))
    
    return result

def f_hour_angle(Gamma, time_UTC, long):
    long_rad = long*2*np.pi/360
    E_t = f_equation_of_time(Gamma)
    
    result = (2*np.pi/24)*(- (time_UTC + long_rad * (24/(2*np.pi)) ) - E_t ) + np.pi
    
    return result

# date_time is a datetime value (or array) (UTC)
# latitude in degrees
# longitude in degrees (positive East)
def f_cos_zenith(date_time, latitude, longitude):
    """
    Compute the cosine of the solar zenith angle
    Input:
       date_time      : date/time variable (time stamp)
       latitude       : geographic latitude (degree)
       longitude      : geographic longitude (degree)
     Return:
       cosine of solar zenith angle
    """
    if (type(date_time) == pd.core.series.Series):
        doy = date_time.dt.dayofyear
        t_UTC = date_time.dt.hour + date_time.dt.minute/60.0
    else:
        my_error("f_cos_zenith: does not know how to deal with date_time variable")

    Gamma = 2*np.pi*(doy-1)/365
    lat_rad = latitude*2*np.pi/360
    long_rad = longitude*2*np.pi/360
    hour_angle = f_hour_angle(Gamma, t_UTC, longitude)
    decl = f_declination (Gamma)
    result = np.sin(decl)*np.sin(lat_rad) + np.cos(decl)*np.cos(lat_rad)*np.cos(hour_angle)
    
    result = np.maximum(0, result)
    return result

def f_ecc_factor(date_time):
    c0 = 1.000110
    c1 = 0.03422
    s1 = 0.001280
    c2 = 0.000719
    s2 = 0.000077
    
    if (type(date_time) == pd.core.series.Series):
        doy = date_time.dt.dayofyear
    else:
        my_error("f_ecc_factor: does not know how to deal with date_time variable")
        
    Gamma = 2*np.pi*(doy-1)/365   
    result = c0 + c1*np.cos(Gamma) + s1*np.sin(Gamma) + \
                  c2*np.cos(2*Gamma) + s2*np.sin(2*Gamma)
    return result

def f_atm_transmissivity(date_time, latitude, longitude, K_in):
    """
    Compute the broadband atmospheric transmissivity
    Input:
       date_time      : date/time variable (time stamp)
       latitude       : geographic latitude (degree)
       longitude      : geographic longitude (degree)
       K_in           : global radiation (W/m2)
    Return:
       broadband atmospheric transmissivity (-)
    """
    I0 = 1365 # (W/m2)
    
    K_0 = I0 * f_ecc_factor(date_time) * f_cos_zenith(date_time, latitude, longitude)
    
    result = K_in/K_0
    
    result = np.where(np.isnan(result), result, np.maximum(result,0.0))
    result = np.where(np.isnan(result), result, np.minimum(result,1.0))
    
    return result

def check_z0(df, z0):
    zu = 10
    karman = 0.4
    my_z0 = zu / np.exp((karman*df['u_10'] / df['ustar_m'] ))
    
    dev = (z0-my_z0)
    my_check = dev.median()/z0.median()
    
    if (abs(my_check) <= 0.1):
        print('Your z0 values seem correct')        
    elif (my_check > 0.1):
        my_error('Your z0 values are probably too high') 
    elif (my_check < -0.1):
        my_error('Your z0 values are probably too low') 
        
        
def check_rc(df_in, rc_in):
    # Compute the canopy resistance 
    # First determine the aerodynamic resistance with the function f_ra
    zu = 10   # m
    zT = 1.5  # m
    d  = 0    # m
    z0 = 0.02  # m (best guess from our own data)
    z0h = 0.1*z0 # m (usually z0h is taken as 0.1 times z0)
    ra = f_ra(df_in['u_10'], zu, zT, d, z0, z0h)

    # Next collect the required other variables (temperature, vapour pressure, net radiation, ....
    # Note that the LvE used in the equation above is the *actual* latent heat flux (i.e. the 
    # eddy-covariance flux, available here as df['LvE_m'])
    T = df_in['T_1_5'] + 273.15
    p = df_in['p']*100
    q = df_in['q']
    s = f_s(T)
    esat = f_esat(T)

    gamma = f_gamma(T, p, q)
    cp    = f_cp(q)
    Qnet = df_in['Q_net_m']
    G    = df_in['G_0_m']
    LvE  = df_in['LvE_m']
    ea   = df_in['e']
    rho  = df_in['rho']

    # Now compute the canopy resistance. To prevent errors it can be helpful to split the 
    # horrible equation in a number of handy chunks.
    numer1 = s*(Qnet - G)
    numer2 = (rho*cp/ra)*(esat - ea)
    my_rc = ra *( (numer1 + numer2)/(gamma*LvE) - (s/gamma) - 1)

    dev = (rc_in-my_rc)
    my_check = dev.median()/my_rc.median()
    print("Mycheck = ", my_check)
    if (abs(my_check) <= 0.05):
        print('Your rc values seem correct')        
    elif (my_check > 0.05):
        my_error('Your rc values are probably too high') 
        print('Note that we assumed that you use variable %s for wind speed, and a value of %f for the roughness length'%('u_10',z0))
    elif (my_check < -0.05):
        my_error('Your rc values are probably too low')
        print('Note that we assumed that you use variable %s for wind speed, and a value of %f for the roughness length'%('u_10',z0))
        
