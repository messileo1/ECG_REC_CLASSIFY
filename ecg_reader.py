import numpy as np
import pywt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def denoise(signal, wavelet='db5', level=9):
    """
    wavelet denoise preprocess using mallat algorithm
    :param signal: 输入信号
    :param wavelet: 小波基
    :param level: 尺度
    :return:
    """
    print('denoise....')
    # 将时域信号进行9尺度变换到频域，小波基选用db5，返回值即为各尺度系数
    coefficients = pywt.wavedec(data=signal, wavelet=wavelet, level=level)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coefficients
    print('\tcD1 length {} ,shape {}'.format(len(cD1), cD1.shape))
    print('\tcD2 length {} ,shape {}'.format(len(cD2), cD2.shape))
    print('\tcD3 length {} ,shape {}'.format(len(cD3), cD3.shape))
    print('\tcD4 length {} ,shape {}'.format(len(cD4), cD4.shape))
    print('\tcD5 length {} ,shape {}'.format(len(cD5), cD5.shape))
    print('\tcD6 length {} ,shape {}'.format(len(cD6), cD6.shape))
    print('\tcD7 length {} ,shape {}'.format(len(cD7), cD7.shape))
    print('\tcD8 length {} ,shape {}'.format(len(cD8), cD8.shape))
    print('\tcD9 length {} ,shape {}'.format(len(cD9), cD9.shape))
    print('\tcA9 length {} ,shape {}'.format(len(cA9), cA9.shape))

    # denoise using soft threshold
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)

    # 去除基线漂移, 9层低频信息
    cA9.fill(0)

    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coefficients) - 2):
        coefficients[i] = pywt.threshold(coefficients[i], threshold)

    # 最后对小波系数进行反变换，获得去噪后的信号
    return pywt.waverec(coeffs=coefficients, wavelet=wavelet)


class ECGReader(FigureCanvas):
    def __init__(self, file_path, number, length = None):
        """
        MIT-BIT 数据集所在的目录
        :param file_path:
        :param number:文件编号
        :param length: 采样点长度
        """
        self.axes = None
        # self.fig 是一个 matplotlib.figure.Figure 对象，代表整个图形
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        FigureCanvas.__init__(self, self.fig)

        self.__file_path = file_path
        self.__number = number
        self.__hex_file = self.__file_path + '/' + self.__number + '.hea'
        self.__data_file = self.__file_path + '/' + number + '.dat'
        self.__atr_file = self.__file_path + '/' + number + '.atr'
        # 信号头文件信息
        self.num_sig = None  # 信号数量
        self.sig_freq = None  # 采样率
        self.dformat = []  # 数据存储格式
        self.gain = []  # 信号增益
        self.bit_res = []  # 位分辨率
        self.zero_value = []  # 零值
        self.first_value = []  # first integer value of signal
        self.lead = []  # 导联
        # 信号数据文件
        self.SAMPLES2READ = length  # 读取的数据长度
        self.data = None  # 数据
        self.time = None  # 采样时间
        # 注释文件
        self.atr_time = []  # 注释对应time
        self.annotd = None  # 注释文本内容

    def load_hea(self):
        """
        读取hea头文件：将源文件按行读取并记录下各参数含义，存入变量。
        :return:
        """
        print("loading the ecg hea of No." + self.__number)

        with open(self.__hex_file, "r") as f:
            # 读取第一行
            line = f.readline()

            # 默认按照空格分隔
            z = line.split()

            # number of signals, sample rate of data， 比如100.hex文件，num_sig=2  sig_freq=360
            self.num_sig, self.sig_freq = int(z[1]), int(z[2])
            print('\tnumber of signals {}, sample rate of data {}'.format(self.num_sig, self.sig_freq))

            # 解析每个信号信息
            for i in range(self.num_sig):
                # 读取新行
                line = f.readline()
                # 默认按照空格分隔
                z = line.split()
                # format; here only 212 is allowed
                self.dformat.append(int(z[1]))
                # number of integers per mV
                self.gain.append(int(z[2]))
                # bit resolution
                self.bit_res.append(int(z[3]))
                # integer value of ECG zero point
                self.zero_value.append(int(z[4]))
                # first integer value of signal (to test for errors)
                self.first_value.append(int(z[5]))
                # 导联
                self.lead.append(z[8])
        print('\tformat:', self.dformat)
        print('\tnumber of integers per mV:', self.gain)
        print('\tbit resolution:', self.bit_res)
        print('\tinteger value of ECG zero point:', self.zero_value)
        print('\tfirst integer value of signal (to test for errors):', self.first_value)
        print('\tlead:', self.lead)

    def load_data(self):
        """
        读取data数据文件：三个字节存储两个信号数据，对信号进行处理，步骤如下
            A 将原始数据转换为SAMPLES2READx3矩阵
            data 由SAMPLES2READx3列矩阵转的SAMPLES2READx2列矩阵
            data[:,0] 第一个信号幅度；
            data[:,1] 第二个信号幅度；
            如果信号量num_sig=2，信号幅度减去零点再除以信号增益
            如果信号量num_sig=1，……
        :return:
        """
        print("loading the ecg dat of No." + self.__number)

        # 以二进制格式读入dat文件
        with open(self.__data_file, "rb") as f:
            # 读取所有数据 比如100.dat 650000*3=1950000
            byte_array = f.read()
            print('\traw data length:', len(byte_array))

            # 根据给定的字节数据和数据类型创建一个Numpy数组，即将读入的二进制文件转化为unit8格式
            raw_data = np.frombuffer(byte_array, dtype=np.uint8)
            # 将原始具转为Nx3矩阵
            A = raw_data.reshape(int(raw_data.shape[0] / 3), 3)[:self.SAMPLES2READ].astype(np.uint32)
            # 未指定长度
            if self.SAMPLES2READ is None:
                self.SAMPLES2READ = A.shape[0]

            # 创建矩阵M，保存数据
            self.data = np.zeros((self.SAMPLES2READ, 2))
            self.data[:, 0] = ((A[:, 1] & 0x0f) << 8) + A[:, 0]
            self.data[:, 1] = ((A[:, 1] & 0xf0) << 4) + A[:, 2]

            if (self.data[1, :] != self.first_value).any():
                print("inconsistency in the first bit values")
                return

            if self.num_sig == 2:
                self.data[:, 0] = (self.data[:, 0] - self.zero_value[0]) / self.gain[0]
                self.data[:, 1] = (self.data[:, 1] - self.zero_value[1]) / self.gain[1]
                self.time = np.linspace(0, self.SAMPLES2READ - 1, self.SAMPLES2READ) / self.sig_freq
            elif self.num_sig == 1:
                M = []
                self.data[:, 0] = self.data[:, 0] - self.zero_value[0]
                self.data[:, 1] = self.data[:, 1] - self.zero_value[1]
                for i in range(self.data.shape[0]):
                    M.append(self.data[:, 0][i])
                    M.append(self.data[:, 1][i])
                M.append(0)
                del M[0]
                self.data = np.array(M) / self.gain[0]
                self.time = np.linspace(0, 2 * self.SAMPLES2READ - 1, 2 * self.SAMPLES2READ) / self.sig_freq
            else:
                print("\tSorting algorithm for more than 2 signals not programmed yet!")

            print('\tReal first value {},unit {}'.format(self.data[0, :], 'mV'))

    def load_atr(self):
        """
        读取atr注释文件
        :return:
        """
        print("loading the ecg atr of No." + self.__number)

        with open(self.__atr_file, "rb") as f:
            # 读取所有数据 比如100.atr
            byte_array = f.read()
            print('\tatr length:', len(byte_array))

            # 根据给定的字节数据和数据类型创建一个Numpy数组，即将读入的二进制文件转化为unit8格式
            raw_data = np.frombuffer(byte_array, dtype=np.uint8)

            # 转换为 Nx2 矩阵
            A = raw_data.reshape(int(raw_data.shape[0] / 2), 2).astype(np.uint32)

            annot = []
            i = 0
            while i < A.shape[0]:
                # 读取高6位
                annoth = A[i, 1] >> 2
                if annoth == 59:
                    annot.append(A[i + 3, 1] >> 2)
                    self.atr_time.append(
                        A[i + 2, 0] + (A[i + 2, 1] << 8) + (A[i + 1, 0] << 16) + (A[i + 1, 1] << 24))
                    i += 3
                elif annoth == 60:
                    pass
                elif annoth == 61:
                    pass
                elif annoth == 62:
                    pass
                elif annoth == 63:
                    hilfe = ((A[i, 1] & 3) << 8) + A[i, 0]
                    hilfe = hilfe + hilfe % 2
                    i += int(hilfe / 2)
                else:
                    self.atr_time.append(((A[i, 1] & 3) << 8) + A[i, 0])
                    annot.append(A[i, 1] >> 2)
                i += 1

            del annot[len(annot) - 1]
            del self.atr_time[len(self.atr_time) - 1]

            self.atr_time = np.array(self.atr_time)
            self.atr_time = np.cumsum(self.atr_time) / self.sig_freq

            ind = np.where(self.atr_time <= self.time[-1])[0]
            self.atr_time = self.atr_time[ind]

            annot = np.round(annot)
            self.annotd = annot[ind]

    def plt_ecg(self):
        """
        绘制心电图
        :return:
        """
        # 调整子图之间的垂直间距
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, hspace=0.7)
        if self.num_sig == 2:
            # self.axes 则是一个 matplotlib.axes.Axes 对象，代表一个子图
            self.axes = self.fig.add_subplot(211)
        else:
            self.axes = self.fig.add_subplot(111)
        self.axes.plot(self.time, self.data[:, 0], linewidth="0.5", c="r")
        self.axes.set_xlim(self.time[0], self.time[-1])
        self.axes.set_xlabel("Time / s")
        self.axes.set_ylabel("Voltage / mV({})".format(self.lead[0]))
        self.axes.set_title("ECG signal")
        for i in range(len(self.atr_time)):
            self.axes.text(self.atr_time[i], 0, str(self.annotd[i]))

        if self.num_sig == 2:
            self.axes = self.fig.add_subplot(212)
            self.axes.plot(self.time, self.data[:, 1], linewidth="0.5", c="b")
            self.axes.set_xlim(self.time[0], self.time[-1])
            self.axes.set_xlabel("Time / s")
            self.axes.set_ylabel("Voltage / mV({})".format(self.lead[1]))
            self.axes.set_title("ECG signal")
            for i in range(len(self.atr_time)):
                self.axes.text(self.atr_time[i], 0, str(self.annotd[i]))
