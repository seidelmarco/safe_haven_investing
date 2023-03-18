import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Plotter:
    """
    The  Plotter class produces matplotlib-figures that are correctly formatted for a certain data analysis
    for a project.
    Ziel ist es, dass ich aus der Klasse Instanzen bilde und jeden Dataframe auf jede Art und Weise mit
    unterschiedlichen Legenden und ax-patches plotten kann.
    """
    def __init__(self, df, rows=1, cols=1, legend=dict(x=0.77, y=1)):
        """
        initialize object attributes and create figure
        the rows and cols default is 1, but can be changed to add
        subplots
        :param df: choose your favorite dataframe for plotting
        :param rows: number of rows of plots
        :param cols: number of columns of plots
        :param legend: initial position
        :param args:
        :param kwargs:
        """
        self.df = df
        self.foods = list(set(df['food']))
        self.mode = 'lines'

        # colors for each item
        self.colors = {
            'apple': 'crimson',
            'avocado': 'darkgreen',
            'blueberry': 'royalblue'
            }

        # markers for each food
        self.marker_dict = {
            'apple': 'square',
            'avocado': 'circle',
            'blueberry': 'x'
            }

        # misc. figure parameters
        self.params = {
            'linewidth': 6,
            'mrkrsize': 10,
            'opacity': 0.8,
            'width': 850,
            'length': 700
        }

        # font for figure labels and legend
        self.lab_dict = dict(
            family='Arial',
            size=26,
            color='black'
            )

        # font for number labeling on axes
        self.tick_dict = dict(
            family='Arial',
            size=24,
            color='black'
            )

        # initialize figure as subplots
        self.fig = make_subplots(rows=rows, cols=cols)

        # general figure formatting
        # set font, borders, size, background color,
        # and legend  position for figure
        self.fig.update_layout(
            font=self.lab_dict,
            margin=dict(r=20, t=20, b=10), # remove white space
            autosize=False,
            width=850,
            height=700,
            plot_bgcolor='black',
            legend=legend
        )



    """
    Then, we add a method to our Plotter class that adds traces to the figure (i.e., plots the data) 
    that we initialized:
    """

    def plot(self, x_col, y_col, row=1, col=1, showlegend=True):
        """
        plot data on Plotly figure for all foods
        x_col = column from dataframe to plot on x-xaxis
        y_col = column from dataframe to plot on y-xaxis
        row/col = which plot the trace should be added to
        showlegend = boolean; show legend on graph
        """
        for food in self.foods:
            x = self.df[x_col].loc[self.df['food'] == food]
            y = self.df[y_col].loc[self.df['food'] == food]

            # add trace to figure
            self.fig.add_trace(go.Scatter(
                x=x,
                y=y,
                showlegend=showlegend,
                mode=self.mode,
                name=food,
                line=dict(
                    width=self.params['linewidth']
                ),
                opacity=self.params['opacity'],
                marker=dict(
                    color=self.colors[food],
                    symbol=self.marker_dict[food],
                    size=self.params['mrkrsize'],
                )
            ),
                row=row,
                col=col
            )

    # Next, we can add methods to the Plotter class that formats both the x and y-axes in a standardized way:
    def update_xaxis(self, xlabel='Time', xlim=[0, 60], row=1, col=1):
        """
        format x-axis by adding axis lines, ticks, etc.
        xlabel = label for x-axis (default is Time (s))
        xlim = range for x-axis (default is 0 to 60)
        row/col = which graph to format
        :param xlabel:
        :param xlim:
        :param row:
        :param col:
        :return:
        """
        self.fig.update_xaxes(
            title_text=xlabel,
            range=xlim,
            showline=True,
            linecolor='black',
            linewidth=2.4,
            showticklabels=True,
            ticks='outside',
            mirror='allticks',
            tickwidth=2.4,
            tickcolor='black',
            tickfont=self.tick_dict,
            row=row,
            col=col
        )

    def update_yaxis(self, ylabel='', ylim=[-1,1],row=1,col=1):
        """
        format y-axis by adding axis lines, ticks, etc.
        ylabel = label for y-axis (default is blank)
        ylim = range for y-axis  (default is -1 to 1)
        row/col = which graph to format
        :param ylabel:
        :param ylim:
        :param row:
        :param col:
        :return:
        """

        self.fig.update_yaxes(
              title_text=ylabel,
              range=ylim,
              showline=True,
              linecolor='black',
              linewidth=2.4,
              showticklabels=True,
              ticks='outside',
              mirror='allticks',
              tickwidth=2.4,
              tickcolor='black',
              tickfont=self.tick_dict,
              row=row,
              col=col
        )


"""
Now that our class is written, the question is “How do we use it?”. We need to create objects from the class, 
add traces to the figure, and then display the figure.

This can be done in one of two ways:

In-script by adding the following code to the end of the script containing your class:
"""

"""Driver-code:"""


def main():
    """
    generate an example dataframe,
    then initialize the Plotter object
    :return:
    """
    foods = ['apple', 'avocado', 'blueberry']

    joined_frames_list = []

    for food in foods:
        time_s = np.linspace(0, 60, 200)
        sin = np.sin(time_s) * (foods.index(food)+1)/3
        cos = np.cos(time_s) * (foods.index(food) + 1) / 3
        temp_df = pd.DataFrame(dict(
            time_s=time_s,
            sin=sin,
            cos=cos
        ))
        temp_df['food'] = food
        joined_frames_list.append(temp_df)
    df = pd.concat(joined_frames_list, ignore_index=True)

    # initialize Plotter object and format axes
    fig = Plotter(df)
    fig.update_xaxis()
    fig.update_yaxis()
    fig.plot('time_s', 'sin')
    fig.fig.show()


if __name__ == '__main__':
    main()












# Todo: for later ... build a standard-class for matplotlib


    def plottype(self, plottype = 'line'):

        match plottype:
            case 'bar':
                self.df.plot()
            case 'barh':
                self.df.plot()
            case 'boxplot':
                self.df.plot()
            case 'scatter':
                self.df.plot()
            case 'candle':
                self.df.plot()
            case default:
                self.df.plot()


    def plotting(self, df, columns: list):
        print(df)

        self.df.plot()
        #df['Adj Close'].plot()
        plt.show()

        df[['High', 'Low']].plot()
        plt.show()




