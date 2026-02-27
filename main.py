import os
import sys
from collections import Counter

import numpy as np
import tensorflow as tf
from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog

from ecg_canvas import DoubleCanvas, Signal, SingleCanvas
from ecg_detect import ECGDetect
from ecg_reader import ECGReader, denoise
# 导入设计的ui界面转换成的py文件
from ui.ui_main import Ui_MainWindow
from ecg_model import SparseFocalLoss, SparseFocalLoss_previous

def clear_layout(layout):
    """
    # 清空布局中的所有部件
    :param layout:
    :return:
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def load_model():
    """
    加载CNN网络
    :return:
    """
    # project root path
    project_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_path, 'best_ecg_model.h5')

    # import the pre-trained model if it exists（导入训练好的模型）
    print('Import the pre-trained model, skip the training process')

    """
    记得加载时要加载训练时用的损失函数
    """
    # 函数1：可对不同类别进行权重加权（如乘10操作）的损失函数
    return tf.keras.models.load_model(filepath=model_path,custom_objects={'SparseFocalLoss': SparseFocalLoss})

    # 函数2：定义适配多分类稀疏标签的Focal Loss类，但没加权
    # return tf.keras.models.load_model(filepath=model_path,custom_objects={'SparseFocalLoss': SparseFocalLoss_previous})
    # # 这是最初的损失函数
    # return tf.keras.models.load_model(filepath=model_path)


class UIMain(QMainWindow):
    """
     UI界面
    """
    # 用于发送要处理的数据
    data_send = pyqtSignal(np.ndarray)

    def __init__(self):
        # QMainWindow构造函数初始化
        super().__init__()
        self.reader = None
        self.interval = 5
        self.read_length = 360 * self.interval
        self.current_index = None
        self.ui = Ui_MainWindow()
        # 这个函数本身需要传递一个MainWindow类，而该类本身就继承了这个，所以可以直接传入self
        self.ui.setupUi(self)
        # 加载CNN网络
        self.model = load_model()
        # 点击按钮打开文件对话框
        self.ui.btn_open_file.clicked.connect(self.open_file_dialog)
        # 初始化布局
        self.orginal_signal_layout = QVBoxLayout()
        self.denoise_layout = QVBoxLayout()
        self.qrs_layout = QVBoxLayout()
        self.p_layout = QVBoxLayout()
        self.t_layout = QVBoxLayout()
        self.ui.gbx_orginal_signal.setLayout(self.orginal_signal_layout)
        self.ui.gbx_denoise.setLayout(self.denoise_layout)
        self.ui.gbx_qrs.setLayout(self.qrs_layout)
        self.ui.gbx_p.setLayout(self.p_layout)
        self.ui.gbx_t.setLayout(self.t_layout)

        # 信号与槽
        self.emitter = MySignalEmitter()  # 信号的发射器
        self.timer = QTimer()  # 创建一个定时器
        self.timer.timeout.connect(self.timer_function)  # 连接定时器的超时信号到函数
        self.emitter.data_send.connect(self.process_data)  # 连接信号与槽

    def open_file_dialog(self):
        """
        打开文件对话框
        :return:
        """
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)

        # 设置过滤器，只显示后缀名为.dat的文件
        file_dialog.setNameFilter("DAT 文件 (*.dat)")

        if file_dialog.exec_():
            # 获取所选文件的路径
            selected_file = file_dialog.selectedFiles()[0]
            print("选择的文件:", selected_file)
            # 获取文件名并去除文件后缀
            file_name = os.path.basename(selected_file)
            file_name_without_extension = os.path.splitext(file_name)[0]
            print("选择的文件名（不带后缀）:", file_name_without_extension)
            self.process(file_name_without_extension)


    def process(self, number):
        # 1. 加载原始信号，并在UI界面显示  采样频率360，每分钟360*60=21600个采样点，每5秒1800个采样点
        self.reader = ECGReader('./st-petersburg-incart-12-lead-arrhythmia-database-1.0.0', number)
        self.reader.load_hea()
        self.reader.load_data()
        self.reader.load_atr()

        # 当前处理数据
        all_data = self.reader.data[:, 0]
        print('读取到数据总长度为：', len(all_data))

        self.current_index = 0

        # 先停止定时器
        if self.timer.isActive():
            self.timer.stop()

        # 首次执行
        self.timer_function()
        # 开启定时器
        self.timer.start(self.interval * 1000)

    def process_data(self, data):
        # 清空布局
        clear_layout(self.orginal_signal_layout)
        clear_layout(self.denoise_layout)
        clear_layout(self.qrs_layout)
        clear_layout(self.p_layout)
        clear_layout(self.t_layout)

        # 绘制波形
        ecg_canvas = DoubleCanvas(
            Signal(data[:, 0], "ECG signal",
                   "sample point",
                   "Voltage / mV({})".format(self.reader.lead[0])),
            Signal(data[:, 1], "ECG signa",
                   "sample point",
                   "Voltage / mV({})".format(self.reader.lead[1])))
        ecg_canvas.plt()
        self.orginal_signal_layout.addWidget(ecg_canvas)

        data = data[:, 0]

        # 2. 去噪，并在UI界面显示信号0去噪前后对比图
        rdata = denoise(data)

        compare_canvas = DoubleCanvas(
            Signal(data, "ECG signal", "sample point", "Voltage / mV"),
            Signal(rdata, "ECG signal after denoise", "sample point", "Voltage / mV"))
        compare_canvas.plt()

        self.denoise_layout.addWidget(compare_canvas)
        # 3. QRS、T、T波检测
        detect = ECGDetect(rdata)
        # 3.1 QRS波查找
        heart_beat, q_loca, r_loca, s_loca = detect.find_qrs()  # heart_beat为此刻的心率

        self.write_heart_beat_data_to_file(heart_beat)
        # 3.2 设置心率值
        self.ui.let_heart_beat.setText(str(int(heart_beat)))

        # 3.3 绘制QRS波形
        qrs_canvas = SingleCanvas(
            Signal([rdata, q_loca, r_loca, s_loca], "ECG signal and QRS", "sample point", "Voltage / mV"))
        qrs_canvas.plt()

        self.qrs_layout.addWidget(qrs_canvas)

        # 3.4 P、T波查找
        p_loca, p_loca_begin, p_loca_end, t_loca, t_loca_begin, t_loca_end = detect.find_pt()

        # 3.5 绘制P波形
        p_canvas = SingleCanvas(
            Signal([rdata, p_loca, p_loca_begin, p_loca_end], "ECG signal and P", "sample point", "Voltage / mV"))
        p_canvas.plt()

        self.p_layout.addWidget(p_canvas)

        # 3.6 绘制T波形
        t_canvas = SingleCanvas(
            Signal([rdata, t_loca, t_loca_begin, t_loca_end], "ECG signal and T", "sample point", "Voltage / mV"))
        t_canvas.plt()

        self.t_layout.addWidget(t_canvas)

        # 4. 预测
        X_test = []
        # 获取QRS波R点所在的索引
        r_index = np.where(r_loca)[0]
        for index in r_index:
            if index >= 99 and (index + 201) <= len(rdata):
                X_test.append(rdata[index - 99:index + 201])
        X_test = np.array(X_test)
        print('测试集', X_test.shape)

        # 进行预测
        Y_test = np.argmax(self.model.predict(X_test), axis=-1)
        counter = Counter(Y_test)
        # 找到出现次数最多的索引及其出现次数
        most_common_index, count = counter.most_common(1)[0]
        ecg_class_set = ['N', 'A', 'V', 'L', 'R']
        print("出现次数最多的索引:", most_common_index)
        print("出现次数:", count)
        Y_test = ecg_class_set[most_common_index]
        print('预测结果', Y_test)
        self.ui.let_class.setText(Y_test)


    def write_heart_beat_data_to_file(self, heart_beat_data):
        try:
            with open("a.txt", "a+", encoding="utf-8") as f:
                data_str = str(heart_beat_data)
                f.write(data_str + '\n')
                f.flush()
        except Exception as e:
            print("写入文件失败")




    def timer_function(self):
        """
        定时任务，每interval秒执行依次
        """
        # 结束索引
        end_index = self.current_index + self.read_length
        data = self.reader.data[self.current_index:end_index:]
        self.emitter.data_send.emit(data)

        # 准备下一轮
        self.current_index = end_index
        while end_index > len(self.reader.data[:, 0]):
            self.timer.stop()


class MySignalEmitter(QObject):
    # 用于发送要处理的数据
    data_send = pyqtSignal(np.ndarray)


if __name__ == '__main__':
    # 先建立一个app
    app = QApplication(sys.argv)
    # 初始化一个对象，调用init函数，已加载设计的ui文件
    ui = UIMain()
    # 显示这个ui
    ui.show()
    # 运行界面，响应按钮等操作
    sys.exit(app.exec_())


