import matplotlib.pyplot as plt
import plotly.graph_objects as go  # plotly must be installed thru the Ipython Console
import plotly.io as pio
import numpy as np


def plot_perf(self, perf, fig_perf):
    # plt.plot(np.arange(i), perf[:i])
    no_epochs = perf.shape[0]
    plt.figure(figsize=fig_perf['figsize'])  # witdth:heigth = 16 : 10
    plt.semilogy(np.arange(no_epochs), perf)
    plt.yscale(fig_perf['y_scale'])

    # font_title = {'fontsize': 18, 'fontweight': 'bold',
    #               'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
    # plt.title(fig_perf['fig_title'], fontdict=font_title)
    font_title = {'fontsize': 18, 'fontweight': 'bold'}
    plt.title(fig_perf['fig_title'], loc='center', **font_title)

    font_axis = {'fontsize': 16, 'fontweight': 'bold'}
    plt.ylabel(fig_perf['y_label'], loc='center', **font_axis)
    #        plt.yticks(fontdict = font_axis)
    plt.xlabel(fig_perf['x_label'], loc='center', **font_axis)
    #        plt.xticks(fontdict = font_axis)
    plt.show()
    # plt.savefig('Training Convergence ' + self.train_method + '.pdf', format='pdf')
    plt.savefig('results/' + fig_perf['file_name'] + '.pdf', format='pdf')


def plotly_perf(perf, fig_data):

    pio.renderers.default = "browser"
    # Default renderers persist for the duration of a single session,
    # but they do not persist across sessions.
    # If you are working in an IPython kernel,
    # this means that default renderers will persist for the life of the kernel,
    # but they will not persist across kernel restarts.
    # plotly.tools.set_credentials_file(username='cuter490703', api_key='cjm8RyxPeAIgJQx3DwMu')
    no_epochs = perf.shape[0]
    x_line = np.linspace(1, no_epochs, no_epochs)
    # Create traces
    # title = 'Training Performance'
    # title = fig_title
    fig_title = fig_data['fig_title']
    y_scale = fig_data['y_scale']
    file_name = fig_data['file_name']
    y_label = fig_data['y_label']
    trace_perf = go.Scatter(x=x_line,
                            y=perf,
                            mode='lines+markers',
                            name=y_label
                            )
    # data = [trace_perf]

    layout = go.Layout(
        title=fig_title,
        legend=dict(font=dict(size=12)),
        xaxis=dict(
            title='epochs',
            # showline=True,
            # showgrid=False,
            # showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            # ticks='outside',
            tickcolor='rgb(204, 204, 204)',
            tickwidth=2,
            ticklen=5,
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title=y_label,
            type=y_scale,
            exponentformat='E',
            # showgrid=False,
            zeroline=True,
            showline=True,
            # showticklabels=True,
        ),
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=True)

    fig = go.Figure(data=[trace_perf], layout=layout)
    # py.plot(data, filename='line-mode')    # onlin plot on plotpy web
    # os.chdir('F:\Courses\勤益科大\Artificial Intelligence\Lectures\Ch 7\Lecture Demo - 2019\Figures')
    # plotly.offline.plot(fig, filename = file_name)     # different plot should have with different file name
    fig.write_html('results/' + file_name)
    fig.show()  # different plot should have with different file name
    # os.chdir('F:\Courses\勤益科大\Artificial Intelligence\Lectures\Ch 7\Lecture Demo - 2019')

#
