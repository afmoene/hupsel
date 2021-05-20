import pandas as pd 
import numpy as np
from numpy import exp
from bokeh.plotting import figure, output_file, show, output_notebook  
from bokeh.palettes import Category10
import itertools
import os # For reading data file

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
        series_list=args
        df_plot = False
        
#    # Check arguments
#    if (type(df)!=pd.DataFrame):
#        print("First argument should be a Dataframe")
#        return

    if (df_plot):
        # Check if units is present as attribute of dataframe
        if ('units' not in df.attrs.keys()):
            units = pd.DataFrame(len(df.keys())*[' '], columns=[df.keys()])
        else:
            units = df.attrs['units']

        # Start plot
        output_notebook()
        if (type(df[series_list[0][0]].values[0]) == np.datetime64):
            xtype = 'datetime'
        else:
            xtype = 'linear'
        p = figure(plot_width=800, plot_height=400, 
                   x_axis_type=xtype, y_axis_type='linear',
                   x_axis_label="%s (%s)"%(series_list[0][0], units[series_list[0][0]][0]), 
                   y_axis_label="%s (%s)"%(series_list[0][1], units[series_list[0][1]][0]))

        # Start color iterator
        color = color_gen()
        # add a line for each series
        for s in series_list:
            for i in range(2):
                if (s[i] not in df.columns):
                    print("Variable %s does not exist in Dataframe"%(s[i]))
                    return
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
                p.line(s[0],s[1], legend_label=s[1], source=df, color=next(color))
            elif (plottype == 'scatter'):
                mycolor = color
                p.scatter(s[0],s[1], legend_label=s[1], source=df, fill_color=next(color))
            elif (plottype == 'bar'):
                barwidth = df[s[0]][1]-df[s[0]][0]
                p.vbar(x=df[s[0]], top=df[s[1]], width=0.9*barwidth, legend_label=s[1], color=color)

        show(p)
    else:
        # We assume that the lists contain data series
        output_notebook()
        print(series_list[0][0].values[0])
        if (type(series_list[0][0].values[0]) == np.datetime64):
            xtype = 'datetime'
        else:
            xtype = 'linear'

        p = figure(plot_width=800, plot_height=400, x_axis_type=xtype)
        # Start color iterator
        color = color_gen()
        # add a line for each series
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

            mycolor = color
            # do plot
            if (plottype == 'line'):
                p.line(s[0],s[1], color=next(color))
            elif (plottype == 'scatter'):
                mycolor = color
                p.scatter(s[0],s[1],  fill_color=next(color))
            elif (plottype == 'bar'):
                barwidth = s[0][1]-s[0][0]
                p.vbar(x=s[0].values, top=s[1], width=0.9*barwidth, color=color)

        # show the results
        show(p)
    
def myreadfile(fname, type='day'):   
    teacher_dir = os.getenv('TEACHER_DIR')
    fullpath = os.path.join(teacher_dir, 'JHL_data', fname)
    fullpath = fname
    # The dataframe that contains the data (both KNMI data and MAQ data)
    df = pd.read_excel(fullpath,skiprows=[0,1,2,3,5,6], sheet_name=0, parse_date=1)
      
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
        
    return df

# Function to compute latent heat of vapourization
# The argument T is assumed to be in Kelvin
# This function is complete and functioning as an example
def Lv_ref(T):
    # Define constants
    c1 = 2501000
    c2 = 0.00095
    c3 = 273.15
    
    # Compute the result
    result =  c1*(1 - c2*(T - c3))
    
    return result   

# Function to compute saturated vapour pressure in Pa
# The argument T is assumed to be in Kelvin
# See secton 7.1 of the AVSi formularium
def esat_ref(T):
    # Define constants (check the values, the zeros are certainly wrong)
    c1 = 611.2
    c2 = 17.62
    c3 = 273.15
    c4 = 30.03
    
    # Compute the result (the structure of the equation is correct)
    result = c1*np.exp((c2*(T-c3))/(-c4+T))
    
    return result

# Function to compute slope of the saturated vapour pressure in Pa/K
# The argument T is assumed to be in Kelvin
# See secton 7.1 of the AVSi formularium
def s_ref(T):
    # Define constants (check the values, the zeros are certainly wrong)
    c1 = 4284
    c2 = 30.03

    # Compute the result (complete the formula)
    result = esat_ref(T)*c1/(-c2+T)**2
    
    return result

# Function to compute the psychrometer constant
# The arguments are temperature T (in K), pressure p in Pa, specific humidity q in kg/kg
def gamma_ref(T, p, q):
    # Define constants (chaeck the values, the zeros are certainly wrong)
    c1 = 65.5
    c2 = 0.84
    c3 = 0.00095
    c4 = 273.15
    c5 = 101300.0

    # Compute the result (complete the formula)
    result = c1*((1+c2*q)/(1-c3*(T-c4)))*(p/c5)
    
    return result  

# Function to compute reference evapotranspiration according to Makkink
# The arguments are global radiation K_in in W/m2, temperature T (in K), pressure p in Pa, specific humidity q in kg/kg
# See secton 7.7 of the AVSI formularium or chapter 7 in Moene & van Dam (2014)
# Please note what is the unit of the resulting number !
def makkink_ref(K_in, T, p, q):
    # First compute s and gamma from the data
    gamma_data = gamma_ref(T, p, q)
    s_data = s_ref(T)
    
    # Now construct the Makkink equation (i.e. replace the '0' by the correct equation)
    # What is the unit?
    result  = 0.65*(s_data/(s_data + gamma_data))*K_in
    
    return result


def checkplot(x, f_ref, f_in, x_name, f_name):
    output_notebook()
    p = figure(x_axis_label="%s"%(x_name), 
               y_axis_label="%s"%(f_name))
    p.line(x,f_ref(x), legend_label='correct %s'%(f_name))
    p.scatter(x,f_in(x), legend_label='your %s'%(f_name))
    show(p)

def check_Lv(Lv_in):
    T = np.linspace(273,273+40)
    ref_data = Lv_ref(T)
    test_data = Lv_in(T)
    rms = np.sqrt(np.mean((ref_data - test_data)**2))
    if (rms < 1e-3*ref_data.mean()):
        print("Well done")
    else:
        print("Not good")
        checkplot(T, Lv_ref, Lv_in, 'T (K)', 'Lv (J/kg)')
        
def check_esat(esat_in):
    T = np.linspace(273,273+40)
    ref_data = esat_ref(T)
    test_data = esat_in(T)
    rms = np.sqrt(np.mean((ref_data - test_data)**2))
    if (rms < 1e-3*ref_data.mean()):
        print("Well done")
    else:
        print("Not good")
        checkplot(T, esat_ref, esat_in, 'T (K)', 'esat (Pa)')
        
def check_s(s_in):
    T = np.linspace(273,273+40)
    ref_data = s_ref(T)
    test_data = s_in(T)
    rms = np.sqrt(np.mean((ref_data - test_data)**2))
    if (rms < 1e-3*ref_data.mean()):
        print("Well done")
    else:
        print("Not good")
        checkplot(T, s_ref, s_in, 'T (K)', 's (Pa/K)')
                
def check_gamma(gamma_in):
    T_range = np.linspace(273,273+40)
    p_range = np.linspace(950e2,1050e2)
    q_range = np.linspace(1e-3,40e-3)
       
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
        ref_data = gamma_ref(var_in[0],var_in[1],var_in[2])    
        test_data = gamma_in(var_in[0],var_in[1],var_in[2])
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
            ref_data = gamma_ref(var_in[0],var_in[1],var_in[2])    
            test_data = gamma_in(var_in[0],var_in[1],var_in[2])

            output_notebook()
            x_name = var_name[i]
            f_name = 'gamma'
            pl = figure(x_axis_label="%s"%(x_name), 
                        y_axis_label="%s"%(f_name))
            pl.line(var[i],ref_data, legend_label='correct %s'%(f_name))
            pl.scatter(var[i],test_data, legend_label='your %s'%(f_name))
            show(pl)

def check_makkink(makkink_in):
    Kin_range = np.linspace(0,900)
    T_range = np.linspace(273,273+40)
    p_range = np.linspace(950e2,1050e2)
    q_range = np.linspace(1e-3,40e-3)
       
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
        ref_data = makkink_ref(var_in[0],var_in[1],var_in[2], var_in[3])    
        test_data = makkink_in(var_in[0],var_in[1],var_in[2], var_in[3])
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
            ref_data = makkink_ref(var_in[0],var_in[1],var_in[2], var_in[3])    
            test_data = makkink_in(var_in[0],var_in[1],var_in[2], var_in[3])

            output_notebook()
            x_name = var_name[i]
            f_name = 'makkink'
            pl = figure(x_axis_label="%s"%(x_name), 
                        y_axis_label="%s"%(f_name))
            pl.line(var[i],ref_data, legend_label='correct %s'%(f_name))
            pl.scatter(var[i],test_data, legend_label='your %s'%(f_name))
            show(pl)
