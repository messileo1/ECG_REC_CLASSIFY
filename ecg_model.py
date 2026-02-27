import datetime
import os

import numpy as np
import seaborn
import tensorflow as tf
import wfdb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ecg_reader import denoise


def get_data_set(number, X_data, Y_data):
    """
    读取心电数据和对应标签,并对数据进行小波去噪
    load the ecg data and the corresponding labels, then denoise the data using wavelet transform
    :param number:
    :param X_data:
    :param Y_data:
    :return:
    """
    ecg_class_set = ['N', 'A', 'V', 'L', 'R']

    # load the ecg data record（读取心电数据记录）
    record = wfdb.rdrecord('./st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/' + number, channel_names=['III'])
    file_path = './st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/' + number
    if not os.path.exists(file_path + '.dat'):
        print(f"警告: 文件 {file_path}.dat 不存在，跳过...")
        return
    else:
        print("loading the ecg data of No." + number)
    data = record.p_signal.flatten()

    # 小波去噪
    rdata = denoise(data)

    # get the positions of R-wave and the corresponding labels（获取心电数据记录中R波的位置和对应的标签）
    annotation = wfdb.rdann('./st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/' + number, 'atr')
    # 获取每一个心拍的R波的尖锋位置的信号点，与心电信号对应
    r_location = annotation.sample
    # 获取R波标注的类别，标注每一个心拍的类型N，L，R等等
    r_class = annotation.symbol

    # remove the unstable data at the beginning and the end（去掉前后的不稳定数据）
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # the data with specific labels (N/A/V/L/R) required in this record are selected, and the others are discarded
    # X_data: data points of length 300 around the R-wave
    # Y_data: convert N/A/V/L/R to 0/1/2/3/4 in order
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            label = ecg_class_set.index(r_class[i])
            x_train = rdata[r_location[i] - 99:r_location[i] + 201]
            X_data.append(x_train)
            Y_data.append(label)
            i += 1
        except ValueError:
            i += 1
    return


def load_data(ratio, random_seed):
    """
    load dataset and preprocess
    加载数据集并进行预处理
    :param ratio:
    :param random_seed:
    :return:
    """


    # # MIT-BIH数据集
    # numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
    #              '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
    #              '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
    #              '231', '232', '233', '234']

    # # 可单独分析
    # numberSet= [ 'I01']

    # INCART数据集
    numberSet= ['I01','I02','I03','I04','I05','I06','I07','I08','I09']
    for i in range(10, 76):
        numberSet.append('I'+str(i))

    data_set = []
    label_set = []
    for n in numberSet:
        get_data_set(n, data_set, label_set)

    print('data_set length ', len(data_set))
    print('label_set length ', len(label_set))
    # 查看心拍的类型
    print('all label class:', wfdb.show_ann_labels())

    # reshape the data and split the dataset（转numpy数组,打乱顺序）
    data_set = np.array(data_set).reshape(-1, 300)
    label_set = np.array(label_set).reshape(-1)

    print('data_set shape ', data_set.shape)
    print('label_set shape ', label_set.shape)


    X_train, X_test, y_train, y_test = train_test_split(data_set, label_set, test_size=ratio, random_state=random_seed, stratify=label_set)
    return X_train, X_test, y_train, y_test


def plot_heat_map(y_test, y_pred):
    """
    confusion matrix
    :param y_test:
    :param y_pred:
    :return:
    """
    con_mat = confusion_matrix(y_test, y_pred)
    # normalize
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # plot
    plt.figure(figsize=(12, 6))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_history_tf(history):
    """
    绘制了训练过程中的模型准确率和损失曲线
    :param history:
    :return:
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()


# build the CNN model
def build_model():
    input_channel = 300 # input_channel需要≥ECG的采样率，以保证覆盖完整采样
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_channel,)),
        # reshape the tensor with shape (batch_size, 300) to (batch_size, 300, 1)
        tf.keras.layers.Reshape(target_shape=(300, 1)),
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 300, 4)
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='relu'),
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 150, 4)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 150, 16)
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='same', activation='relu'),
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 75, 16)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 75, 32)
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='same', activation='relu'),
        # the third pooling layer, average pooling, pooling size=3, stride=2, output shape (batch_size, 38, 32)
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='same'),
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 38, 64)
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='same', activation='relu'),
        # flatten layer, for the next fully connected layer, output shape (batch_size, 38*64)
        tf.keras.layers.Flatten(),
        # fully connected layer, 128 nodes, output shape (batch_size, 128)
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer, dropout rate = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return new_model

# 定义适配多分类稀疏标签的Focal Loss类
class SparseFocalLoss_previous(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, reduction=tf.keras.losses.Reduction.AUTO, name="sparse_focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        y_true = tf.cast(y_true, tf.int32)
        y_pred_true = tf.gather(y_pred, y_true, axis=-1, batch_dims=1)

        p_t = y_pred_true
        ce_loss = -tf.math.log(p_t)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha = tf.gather(self.alpha, y_true, axis=-1)
            focal_loss = alpha * focal_loss

        return focal_loss


# 可对不同类别进行权重加权的损失函数
class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, reduction=tf.keras.losses.Reduction.AUTO, name="sparse_focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha  # 这里传入类别权重
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        y_true = tf.cast(y_true, tf.int32)
        y_pred_true = tf.gather(y_pred, y_true, axis=-1, batch_dims=1)

        p_t = y_pred_true
        ce_loss = -tf.math.log(p_t)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)

        # 核心：应用类别权重（alpha），给第1类乘10
        if self.alpha is not None:
            # 将alpha转为tensor并确保和输入同设备
            alpha = tf.convert_to_tensor(self.alpha, dtype=tf.float32)
            # 根据真实标签y_true，为每个样本匹配对应类别的权重
            alpha = tf.gather(alpha, y_true, axis=-1)
            focal_loss = alpha * focal_loss

        return focal_loss


# 可以对少样本，难分类的进行权重损失×10的操作
class_weights = [0.5, 10.0, 1.0, 1.0, 1.0]
# model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# 初始化带权重的Focal Loss
focal_loss = SparseFocalLoss(alpha=class_weights, gamma=2.0)  # 实例化自定义损失类


# GPU配置代码 强制使用GPU训练，有两种方式

# 方法1. 设置仅在GPU可用时才使用GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存自增长（避免一次性占满显存）
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"物理GPU数量: {len(gpus)}, 逻辑GPU数量: {len(logical_gpus)}")
        print("将使用GPU训练")
    except RuntimeError as e:
        print(e)
else:
    print("未检测到GPU，将使用CPU训练")

# 方法2. 强制指定使用第0块GPU（可选）
# with tf.device('/GPU:0'):
#     model = build_model()  # 把模型构建放在device上下文里

def main():
    # project root path
    project_path = os.path.dirname(os.path.abspath(__file__))
    # define log directory
    # must be a subdirectory of the directory specified when starting the web application
    # it is recommended to use the date time as the subdirectory name
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(project_path, 'logs', current_time)  # 这样
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_path = os.path.join(project_path, 'ecg_model.h5')
    best_model_path = os.path.join(project_path, "best_ecg_model.h5")


    # 训练参数的设置
    # the ratio of the test set （原始值为0.3）
    RATIO = 0.3
    # the random seed （原始值分别为42,128,30）
    RANDOM_SEED = 42
    BATCH_SIZE = 128
    NUM_EPOCHS = 50

    # 是否加载预训练权重
    pre_train = True
    # 是训练还是验证 True-训练，False-验证，预测时记得改回来
    train =  True
    # 是否用最佳模型进行预测，建议选TRUE
    use_best_model = True

    # X_train,y_train is the training set（X_train,y_train为所有的数据集和标签集）
    # X_test,y_test is the test set（# X_test,y_test为拆分的测试集和标签集）
    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED) # RANDOM_SEED

    print('X_train shape ', X_train.shape)
    print('X_test shape ', X_test.shape)
    print('y_train shape ', y_train.shape)
    print('y_train shape ', y_test.shape)

    if os.path.exists(model_path) and not train:
        # import the pre-trained model if it exists（导入训练好的模型）
        print('Import the trained model, skip the training process')
        # model = tf.keras.models.load_model(filepath=model_path, custom_objects={'SparseFocalLoss': SparseFocalLoss_previous})
        if use_best_model == False:
            model = tf.keras.models.load_model(filepath=model_path)
        # 加载最优模型进行预测
        else :
            print("\n加载最优模型进行预测...")
            model = tf.keras.models.load_model(filepath=best_model_path,
                                                custom_objects={'SparseFocalLoss': SparseFocalLoss,
                                                                'SparseFocalLoss_previous': SparseFocalLoss_previous})
    else:
        # build the CNN model（构建CNN模型）
        print(tf.config.list_physical_devices('GPU'))
        model = build_model()
        if pre_train:
            # model = tf.keras.models.load_model(filepath=model_path)
            model = tf.keras.models.load_model(filepath=model_path, custom_objects={'SparseFocalLoss': SparseFocalLoss_previous})


        # 配置模型训练时的损失函数不用关注之前的预训练权重所用的损失函数
        opt = tf.keras.optimizers.legacy.Adam(learning_rate= 0.00001)
        model.compile(# optimizer='adam', # 默认学习率
                      optimizer= opt,
                      # loss='sparse_categorical_crossentropy',
                      loss = focal_loss, # 使用自定义的损失函数
                      metrics=['accuracy'])
        model.summary()

        # 定义回调函数，用于保存最佳模型
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,  # 最优模型保存路径
            monitor='val_loss',  # 监控验证集loss（也可以选loss监控训练集）
            verbose=1,  # 打印保存日志（1=打印，0=不打印）
            save_best_only=True,  # 只保存最优模型（loss最小）
            save_weights_only=False,  # 保存完整模型（False），而非仅权重（True）
            mode='min',  # min=监控指标越小越好（loss），max=越大越好（accuracy）
            save_freq='epoch'  # 每个epoch结束后检查
        )


        # define the TensorBoard callback object（定义TensorBoard对象）
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # train and evaluate model（训练与验证）
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback, checkpoint_callback]) # 添加checkpoint回调

        # save the model，可根据loss或者acc保存最佳模型
        model.save(filepath=model_path)
        print(f"最后一轮模型已保存至: {model_path}")
        print(f"最优模型已保存至: {best_model_path}")


        # plot the training history
        plot_history_tf(history)

    # predict the class of test data
    # y_pred = model.predict_classes(X_test)  # predict_classes has been deprecated
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    # plot confusion matrix heat map
    plot_heat_map(y_test, y_pred)

    """
    各类别的准确率及训练样本量如下：
    0-99.75%   21448
    1-90.34%   611  样本量较少，可以考虑focal loss去压制难样本
    2-98.15%   2112
    3-99.59%   1955
    4-99.09%   1532
    """



if __name__ == '__main__':
    main()
