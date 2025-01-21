import plotly.express as px
import plotly.io as pio


def plot(df, log=False):
    pio.renderers.default = "browser"
    fig = px.line(
        df[
            [
                (df.index[k].hour == 0) & (df.index[k].minute == 0)
                for k in range(len(df.index))
            ]
        ],
        log_y=log,
    )
    fig.show()


def plot_all(df, log=False):
    pio.renderers.default = "browser"
    fig = px.line(df, log_y=log)
    fig.show()
