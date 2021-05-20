import pandas as pd 
import numpy as np
from numpy import exp
from bokeh.plotting import figure, output_file, show, output_notebook  
from bokeh.palettes import Category10
import itertools

# Function to cycle plotting colors (see https://stackoverflow.com/questions/39839409/when-plotting-with-bokeh-how-do-you-automatically-cycle-through-a-color-pallett) 
def color_gen():
    yield from itertools.cycle(Category10[10])


def myplot(df, *args):
    # Check arguments
    if (type(df)!=pd.DataFrame):
        print("First argument should be a Dataframe")
        return

    # Copy list of series (assuming those are the only arguments)
    series_list=args
    
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
    
def myreadfile(fname, type='day'):
    # The dataframe that contains the data (both KNMI data and MAQ data)
    df = pd.read_excel(fname,skiprows=[0,1,2,3,5,6], sheet_name=0, parse_date=1)
      
    # Add the units (read from row 5) as an attribute to the dataframe
    units = pd.read_excel(fname,skiprows=[0,1,2,3], nrows=1) 
    units_dict = {}
    for i in range(len(units.values[0])):
        units_dict[df.keys()[i]] = units.values[0][i]
        df.attrs['units']=units_dict
        
    # Add description of variables
    descr = pd.read_excel(fname,skiprows=[0,1,2,3,5], nrows=1) 
    descr_dict = {}
    for i in range(len(descr.values[0])):
        descr_dict[df.keys()[i]] = descr.values[0][i]
        df.attrs['description']=descr_dict    
        
    return df