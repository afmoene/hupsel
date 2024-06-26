import pandas as pd 
import numpy as np
from numpy import exp
from bokeh.plotting import figure, output_file, show, output_notebook, ColumnDataSource
from bokeh.palettes import Category10
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
good_range_u = [1e-3,50]         # m/s -> to prevent division by zero in ra
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
    
    Additional documentation/exanples: 
    * plot one series as a line: `myplot(df,['Date','K_in'])`. 
      The x-name and y-name are given as a list (enclosed in square brackets).
    * plot two series as lines: `myplot(df,['Date','K_in'], ['Date','K_out_m'])`. 
      The second series is also given as a list, separated from the first list with comma.
    * plot a series with something other than a line 
       * `myplot(df,['Date','K_in','-'])`: plot a line
       * `myplot(df,['Date','K_in','o'])`: plot dots
       * `myplot(df,['Date','prec','#'])`: bar graph (only one series per graph)
    * you can also plot series without using a dataframe (assume x, y and z are  arrays): `myplot([x,y],[x,z])`
 

    Parameters
    ----------
    
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
    for key, value in kwargs.items():
        if (key == 'xlabel'):
            my_xlabel = str(value)
        elif (key == 'ylabel'):
            my_ylabel = str(value)
        else:
            my_warning("myplot: unknown keyword argument:%s"%(key))
        
    # Check if more than one bar graph is asked for
    nbar = 0
    for serie in series_list:
        if (len(serie)>2):
            if (serie[2] == '#'):
                nbar += 1
    if (nbar > 1):
        print("You ask for more than one bar graph. We cannot handle that yet.")
        return
    

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
        else:
            xtype = 'linear'
        output_notebook()
        if (not my_xlabel):
            my_xlabel = "%s (%s)"%(series_list[0][0], units[series_list[0][0]])
        if (not my_ylabel):
            my_ylabel = "%s (%s)"%(series_list[0][1], units[series_list[0][1]])
                  
        p = figure(plot_width=800, plot_height=400, 
                   x_axis_type=xtype, y_axis_type='linear',
                   x_axis_label=my_xlabel, 
                   y_axis_label=my_ylabel)

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
                p.scatter(df[s[0]],df[s[1]], legend_label=series_label, fill_color=next(color))
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
        else:
            xtype = 'linear'

        output_notebook()
        p = figure(plot_width=800, plot_height=400, x_axis_type=xtype,
                   x_axis_label=my_xlabel, 
                   y_axis_label=my_ylabel)
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
                p.scatter(s[0],s[1], legend_label=series_label, fill_color=next(color))
            elif (plottype == 'bar'):
                barwidth = s[0][1]-s[0][0]
                p.vbar(x=s[0].values, top=s[1].values, legend_label=series_label, width=0.9*barwidth, color=next(color))

        # show the results
        show(p)
    
def myreadfile(fname, type='day'):   
    #teacher_dir = os.getenv('TEACHER_DIR')
    # fullpath = os.path.join(teacher_dir, 'JHL_data', fname)
    fullpath = fname
    # The dataframe that contains the data (both KNMI data and MAQ data)
    df = pd.read_excel(fullpath,skiprows=[0,1,2,3,5,6], sheet_name=0, parse_dates=[1])
      
    # Add the units (read from row 5) as an attribute to the dataframe
    units = pd.read_excel(fullpath,skiprows=[0,1,2,3], nrows=1) 
    units_dict = {}
    for i in range(len(units.values[0])):
        units_dict[df.keys()[i]] = units.values[0][i]
        df.attrs['units']=units_dict
        
    # Add description of variables
    descr = pd.read_excel(fullpath,skiprows=[0,1,2,3,5], nrows=1) 
    descr_dict = {}
    for i in range(len(descr.values[0])):
        descr_dict[df.keys()[i]] = descr.values[0][i]
        df.attrs['description']=descr_dict    

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
    # make the input variables arrays to ensure that .all() works, even if the input data is a scalar
    if (not ((good_range_u[0] <= np.array(u)) & (np.array(u) <= good_range_u[1] )).all()):
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

    return f_ra(u, zu, zT, d, z0, z0h)

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
