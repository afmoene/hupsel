import pandas as pd 
import numpy as np
from numpy import exp
from bokeh.plotting import figure, output_file, show, output_notebook, ColumnDataSource
from bokeh.palettes import Category10
import itertools
import os # For reading data file

# Define reasonable ranges for meteo data
good_range_T = [273-40,273+50]
good_range_q = [1e-4,1e-1]
good_range_p = [900e2,1100e2]
good_range_Kin = [0,1365]

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


def myplot(*args):
    if (type(args[0]) == pd.DataFrame):
        df = args[0]
        series_list = args[1:]
        df_plot = True
    else:
        # Copy list of series (assuming those are the only arguments)
        series_list = args
        df_plot = False
    
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
        p = figure(plot_width=800, plot_height=400, 
                   x_axis_type=xtype, y_axis_type='linear',
                   x_axis_label="%s (%s)"%(series_list[0][0], units[series_list[0][0]]), 
                   y_axis_label="%s (%s)"%(series_list[0][1], units[series_list[0][1]]))

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
            # do plot
            if (plottype == 'line'):
                p.line(df[s[0]],df[s[1]], legend_label=s[1], color=next(color))
            elif (plottype == 'scatter'):
                mycolor = color
                p.scatter(df[s[0]],df[s[1]], legend_label=s[1], fill_color=next(color))
            elif (plottype == 'bar'):
                barwidth = df[s[0]][1]-df[s[0]][0]
                p.vbar(x=df[s[0]], top=df[s[1]], width = 0.3*barwidth, \
                       legend_label=s[1], color=next(color))

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
        p = figure(plot_width=800, plot_height=400, x_axis_type=xtype)
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
            mycolor = color
            
            # do plot
            if (plottype == 'line'):
                p.line(s[0],s[1], color=next(color))
            elif (plottype == 'scatter'):
                mycolor = color
                p.scatter(s[0],s[1],  fill_color=next(color))
            elif (plottype == 'bar'):
                barwidth = s[0][1]-s[0][0]
                p.vbar(x=s[0].values, top=s[1].values, width=0.9*barwidth, color=next(color))

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
    if (not ((good_range_T[0] < np.array(T)) & (np.array(T) < good_range_T[1] )).all()):
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
    if (not ((good_range_T[0] < np.array(T)) & (np.array(T) < good_range_T[1] )).all()):
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
    if (not ((good_range_T[0] < np.array(T)) & (np.array(T) < good_range_T[1] )).all()):
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
    if (not ((good_range_T[0] < np.array(T)) & (np.array(T) < good_range_T[1] )).all()):
        my_warning("f_gamma: are you sure that the units of your temperature data are correct?")
    if (not ((good_range_p[0] < np.array(p)) & (np.array(p) < good_range_p[1] )).all()):
        my_warning("f_gamma: are you sure that the units of your pressure data are correct?")
    if (not ((good_range_q[0] < np.array(q)) & (np.array(q) < good_range_q[1] )).all()):
        my_warning("f_gamma: are you sure that the units of your specific humidity data are correct?")
    return f_gamma_ref(T, p, q)
    
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
    if (not ((good_range_Kin[0] < np.array(K_in)) & (np.array(K_in) < good_range_Kin[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your global radiation data are correct?")
    if (not ((good_range_T[0] < np.array(T)) & (np.array(T) < good_range_T[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your temperature data are correct?")
    if (not ((good_range_p[0] < np.array(p)) & (np.array(p) < good_range_p[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your pressure data are correct?")
    if (not ((good_range_q[0] < np.array(q)) & (np.array(q) < good_range_q[1] )).all()):
        my_warning("f_makkink: are you sure that the units of your specific humidity data are correct?")
    return f_makkink_ref(K_in, T, p, q)

def checkplot(x, f_ref, f_in, x_name, f_name):
    output_notebook()
    p = figure(x_axis_label="%s"%(x_name), 
               y_axis_label="%s"%(f_name))
    p.line(x,f_ref(x), legend_label='correct %s'%(f_name))
    p.scatter(x,f_in(x), legend_label='your %s'%(f_name))
    show(p)

def check_Lv(f_Lv_in):
    T = np.linspace(good_range_T[0],good_range_T[1])
    ref_data = f_Lv_ref(T)
    test_data = f_Lv_in(T)
    rms = np.sqrt(np.mean((ref_data - test_data)**2))
    if (rms < 1e-3*ref_data.mean()):
        print("Well done")
    else:
        print("Not good")
        checkplot(T, f_Lv_ref, f_Lv_in, 'T (K)', 'Lv (J/kg)')
        
def check_esat(f_esat_in):
    T = np.linspace(good_range_T[0],good_range_T[1])
    ref_data = f_esat_ref(T)
    test_data = f_esat_in(T)
    rms = np.sqrt(np.mean((ref_data - test_data)**2))
    if (rms < 1e-3*ref_data.mean()):
        print("Well done")
    else:
        print("Not good")
        checkplot(T, f_esat_ref, f_esat_in, 'T (K)', 'esat (Pa)')
        
def check_s(f_s_in):
    T = np.linspace(good_range_T[0],good_range_T[1])
    ref_data = f_s_ref(T)
    test_data = f_s_in(T)
    rms = np.sqrt(np.mean((ref_data - test_data)**2))
    if (rms < 1e-3*ref_data.mean()):
        print("Well done")
    else:
        print("Not good")
        checkplot(T, f_s_ref, f_s_in, 'T (K)', 's (Pa/K)')
                
def check_gamma(f_gamma_in):
    T_range = np.linspace(good_range_T[0],good_range_T[1])
    p_range = np.linspace(good_range_p[0],good_range_p[1])
    q_range = np.linspace(good_range_q[0],good_range_q[1])
       
    var=[T_range,p_range,q_range]
    var_name = ['T', 'p', 'q']
    error=[]
    for i in range(len(var)):
        var_in = []
        for j in range(len(var)):
            if (j==i):
                var_in.append(var[j])
            else:
                var_in.append(var[j].mean())
        ref_data = f_gamma_ref(var_in[0],var_in[1],var_in[2])    
        test_data = f_gamma_in(var_in[0],var_in[1],var_in[2])
        rms = np.sqrt(np.mean((ref_data - test_data)**2))
        if (rms < 1e-3*ref_data.mean()):
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
            ref_data = f_gamma_ref(var_in[0],var_in[1],var_in[2])    
            test_data = f_gamma_in(var_in[0],var_in[1],var_in[2])

            output_notebook()
            x_name = var_name[i]
            f_name = 'gamma'
            pl = figure(x_axis_label="%s"%(x_name), 
                        y_axis_label="%s"%(f_name))
            pl.line(var[i],ref_data, legend_label='correct %s'%(f_name))
            pl.scatter(var[i],test_data, legend_label='your %s'%(f_name))
            show(pl)

def check_makkink(f_makkink_in):
    Kin_range =  np.linspace(good_range_Kin[0],good_range_Kin[1])
    T_range = np.linspace(good_range_T[0],good_range_T[1])
    p_range = np.linspace(good_range_p[0],good_range_p[1])
    q_range = np.linspace(good_range_q[0],good_range_q[1])
       
    var=[Kin_range, T_range,p_range,q_range]
    var_name = ['K_in','T', 'p', 'q']
    error=[]
    for i in range(len(var)):
        var_in = []
        for j in range(len(var)):
            if (j==i):
                var_in.append(var[j])
            else:
                var_in.append(var[j].mean())
        ref_data = f_makkink_ref(var_in[0],var_in[1],var_in[2], var_in[3])    
        test_data = f_makkink_in(var_in[0],var_in[1],var_in[2], var_in[3])
        rms = np.sqrt(np.mean((ref_data - test_data)**2))
        if (rms < 1e-3*ref_data.mean()):
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
            ref_data = f_makkink_ref(var_in[0],var_in[1],var_in[2], var_in[3])    
            test_data = f_makkink_in(var_in[0],var_in[1],var_in[2], var_in[3])

            output_notebook()
            x_name = var_name[i]
            f_name = 'makkink'
            pl = figure(x_axis_label="%s"%(x_name), 
                        y_axis_label="%s"%(f_name))
            pl.line(var[i],ref_data, legend_label='correct %s'%(f_name))
            pl.scatter(var[i],test_data, legend_label='your %s'%(f_name))
            show(pl)
