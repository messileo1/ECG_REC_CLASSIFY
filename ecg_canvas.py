from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Signal:
    def __init__(self, signal, title=None, x_label=None, y_label=None, xlim_start=None):
        """
        信号参数
        """
        self.signal = signal
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.xlim_start = xlim_start


class DoubleCanvas(FigureCanvas):
    """
    绘制两个子图
    """

    def __init__(self, signal1: Signal, signal2: Signal):
        # self.fig 是一个 matplotlib.figure.Figure 对象，代表整个图形
        self.axes = None
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.signal1 = signal1
        self.signal2 = signal2
        FigureCanvas.__init__(self, self.fig)

    def plt(self):
        """
        绘制两个信号的波形
        :return:
        """
        # 调整子图之间的垂直间距
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, hspace=0.7)

        # self.axes 则是一个 matplotlib.axes.Axes 对象，代表一个子图
        self.axes = self.fig.add_subplot(211)
        self.axes.plot(self.signal1.signal, linewidth="0.5", c="r")
        if self.signal1.xlim_start is None:
            self.axes.set_xlim(0, len(self.signal1.signal))
        else:
            self.axes.set_xlim(self.signal1.xlim_start, self.signal1.xlim_start + len(self.signal1.signal))
        self.axes.set_xlabel(self.signal1.x_label)
        self.axes.set_ylabel(self.signal1.y_label)
        self.axes.set_title(self.signal1.title)

        self.axes = self.fig.add_subplot(212)
        self.axes.plot(self.signal2.signal, linewidth="0.5", c="b")
        if self.signal2.xlim_start is None:
            self.axes.set_xlim(0, len(self.signal2.signal))
        else:
            self.axes.set_xlim(self.signal2.xlim_start, self.signal2.xlim_start + len(self.signal2.signal))
        self.axes.set_xlabel(self.signal2.x_label)
        self.axes.set_ylabel(self.signal2.y_label)
        self.axes.set_title(self.signal2.title)


class SingleCanvas(FigureCanvas):
    """
    绘制1个子图，包含4组数据
    """

    def __init__(self, signal: Signal):
        # self.fig 是一个 matplotlib.figure.Figure 对象，代表整个图形
        self.axes = None
        self.fig = plt.Figure(figsize=(16, 6), dpi=100)
        self.signal = signal
        FigureCanvas.__init__(self, self.fig)

    def plt(self):
        """
        绘制两个信号的波形
        :return:
        """
        # 调整子图之间的垂直间距
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, hspace=0.7)

        # self.axes 则是一个 matplotlib.axes.Axes 对象，代表一个子图
        self.axes = self.fig.add_subplot(111)
        self.axes.plot(self.signal.signal[0])
        self.axes.plot(self.signal.signal[1], linewidth="0.5", c="black")
        self.axes.plot(self.signal.signal[2], linewidth="0.5", c="red")
        self.axes.plot(self.signal.signal[3], linewidth="0.5", c="black")
        self.axes.set_title(self.signal.title)
        self.axes.set_xlabel(self.signal.x_label)
        self.axes.set_ylabel(self.signal.y_label)
