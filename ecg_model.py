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


def get_data_set(number, X_data, Y_data, base_dir="european-st-t-database-1.0.0"):
    """
    :param number: 数字编号（如"105"）
    :param X_data: 存储ECG数据的列表
    :param Y_data: 存储标签的列表
    :param base_dir: 数据集根文件夹
    """
    ecg_class_set = ['N', 'A', 'V', 'L', 'R']


    file_prefix = f"e0{number}"  # 105 → e0105
    file_path = os.path.join(base_dir, file_prefix)  # 完整路径：european-st-t-database-1.0.0/e0105

    # 检查.dat文件是否存在
    dat_file = f"{file_path}.dat"
    if not os.path.exists(dat_file):
        print(f"警告: 文件 {dat_file} 不存在，跳过...")
        return
    print(f"加载ECG数据: {file_prefix} (路径: {dat_file})")

    # 读取ECG记录（指定V5导联，欧洲ST-T数据集的核心导联）
    try:
        daolian = wfdb.rdrecord(file_path)
        print(daolian.sig_name)
        record = wfdb.rdrecord(file_path, channel_names=['MLII', 'V5','aVF','III','II','MLI', 'MLIII'])
        if record is None or record.p_signal is None or len(record.p_signal) == 0:
            print(f"警告: {file_prefix} 数据读取失败或无有效信号，跳过...")
            return
    except Exception as e:
        print(f"错误: 读取 {file_prefix} 时异常 - {str(e)}，跳过...")
        return

    # 数据预处理：展平+去噪
    data = record.p_signal.flatten()
    rdata = denoise(data)

    # 读取标注文件（.atr）
    try:
        annotation = wfdb.rdann(file_path, 'atr')
    except Exception as e:
        print(f"错误: 读取 {file_prefix}.atr 标注文件异常 - {str(e)}，跳过...")
        return

    # 提取R波位置和标签
    r_location = annotation.sample # 获取每一个心拍的R波的尖锋位置的信号点，与心电信号对应
    r_class = annotation.symbol # 获取R波标注的类别，标注每一个心拍的类型N，L，R等等

    # 去掉前后不稳定数据
    # 因为只选择NAVLR五种心电类型, 所以要选出该条记录中所需要的那些带有特定标签的数据, 舍弃其余标签的点
    start = 10
    end = 5
    i = start
    j = len(r_class) - end

    # 截取R波前后300个点的片段，并过滤标签
    while i < j:
        try:
            # 仅保留NAVLR标签，转换为0-4的数字标签
            label = ecg_class_set.index(r_class[i])
            # 截取R波前99点 + 后201点 = 300点
            x_train = rdata[r_location[i] - 99:r_location[i] + 201]

            # 过滤长度不足300的片段（避免后续reshape报错）
            if len(x_train) != 300:
                i += 1
                continue

            X_data.append(x_train)
            Y_data.append(label)
            i += 1
        except ValueError:
            # 非NAVLR标签，跳过
            i += 1
    return


def load_data(ratio=0.3, random_seed=42, base_dir="european-st-t-database-1.0.0"):
    """
    加载欧洲ST-T数据集并预处理
    :param ratio: 测试集比例
    :param random_seed: 随机种子
    :param base_dir: 数据集根文件夹
    :return: X_train, X_test, y_train, y_test
    """
    # -------------------------- 欧洲ST-T数据集的有效编号列表 --------------------------
    # 替换为你实际拥有的文件编号（可根据之前找文件的脚本输出修改）
    numberSet = [
        '103', '104', '105', '106', '107', '108', '109', '111', '112', '113',
        '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
        '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
        '213', '214', '215', '217', '219', '220', '221', '222', '223', '228',
        '230', '231', '232', '233', '234'
    ]
    # 测试用：仅加载105（注释掉上面的列表，启用此行）
    # numberSet = ['105']

    # 加载数据
    data_set = []
    label_set = []
    for n in numberSet:
        get_data_set(n, data_set, label_set, base_dir)

    # 检查是否加载到有效数据
    if len(data_set) == 0:
        print("错误: 未加载到任何有效ECG数据！")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 转换为numpy数组并reshape
    data_set = np.array(data_set).reshape(-1, 300)
    label_set = np.array(label_set).reshape(-1)

    print(f"\n数据加载完成:")
    print(f"- 总样本数: {len(data_set)}")
    print(f"- 数据形状: {data_set.shape}")
    print(f"- 标签形状: {label_set.shape}")
    print(f"- 标签分布: {np.bincount(label_set)} (对应N/A/V/L/R)")

    counts = np.bincount(label_set, minlength=5)
    ecg_class_set = ['N', 'A', 'V', 'L', 'R']
    print("\n标签分布详情：")
    for i, cls in enumerate(ecg_class_set):
        print(f"{cls} 类数量: {counts[i]}")



    # 划分训练集/测试集（分层抽样，保证标签分布一致）
    X_train, X_test, y_train, y_test = train_test_split(
        data_set, label_set,
        test_size=ratio,
        random_state=random_seed,
        stratify=label_set
    )

    # 标准化（Z-score）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# -------------------------- 以下代码保持不变（模型/训练/可视化） --------------------------
def plot_heat_map(y_test, y_pred):
    """绘制混淆矩阵"""
    if len(y_test) == 0 or len(y_pred) == 0:
        print("警告: 无测试数据，无法绘制混淆矩阵")
        return
    con_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 6))
    seaborn.heatmap(con_mat, annot=True, fmt='d', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (N/A/V/L/R → 0/1/2/3/4)')
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_history_tf(history):
    """绘制训练曲线（准确率+损失）"""
    # 准确率曲线
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

    # 损失曲线
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


def build_model(input_channel=300):
    """构建1D-CNN模型"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_channel,)),
        tf.keras.layers.Reshape(target_shape=(300, 1)),

        # 卷积层1 + 最大池化
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),

        # 卷积层2 + 最大池化
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),

        # 卷积层3 + 平均池化
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='same', activation='relu'),
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='same'),

        # 卷积层4
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='same', activation='relu'),

        # 全连接层
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(5, activation='softmax')  # 5类：N/A/V/L/R
    ])
    return model


# 带类别权重的Focal Loss（适配稀疏标签）
class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=None, gamma=2.0, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
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
            alpha = tf.convert_to_tensor(self.alpha, dtype=tf.float32)
            alpha = tf.gather(alpha, y_true, axis=-1)
            focal_loss = alpha * focal_loss

        return focal_loss


# GPU配置
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU配置成功: 物理GPU={len(gpus)}, 逻辑GPU={len(tf.config.list_logical_devices('GPU'))}")
        except RuntimeError as e:
            print(f"GPU配置失败: {e}")
    else:
        print("未检测到GPU，使用CPU训练")


def main():
    # 1. 初始化配置
    setup_gpu()
    project_path = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(project_path, 'logs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    # 模型保存路径
    model_path = os.path.join(project_path, 'ecg_stt_model.h5')
    best_model_path = os.path.join(project_path, 'best_ecg_stt_model.h5')

    # 2. 训练参数
    RATIO = 0.3  # 测试集比例
    RANDOM_SEED = 42  # 随机种子
    BATCH_SIZE = 128  # 批次大小
    NUM_EPOCHS = 50  # 训练轮数
    pre_train = False  # 是否加载预训练模型（首次训练设为False）
    train = True  # True=训练，False=仅预测
    use_best_model = True  # 是否使用最优模型预测

    # 3. 加载数据（核心：指定欧洲ST-T数据集路径）
    X_train, X_test, y_train, y_test = load_data(
        ratio=RATIO,
        random_seed=RANDOM_SEED,
        base_dir="european-st-t-database-1.0.0"  # 数据集根文件夹
    )
    if len(X_train) == 0:
        print("错误: 无有效数据，程序退出")
        return

    # 4. 构建/加载模型
    if os.path.exists(model_path) and not train:
        # 加载预训练模型
        print(f"加载预训练模型: {model_path}")
        if use_best_model and os.path.exists(best_model_path):
            model = tf.keras.models.load_model(
                best_model_path,
                custom_objects={'SparseFocalLoss': SparseFocalLoss}
            )
        else:
            model = tf.keras.models.load_model(model_path)
    else:
        # 构建新模型
        model = build_model()
        # 加载预训练权重（如果有）
        if pre_train and os.path.exists(model_path):
            print(f"加载预训练权重: {model_path}")
            model.load_weights(model_path)

        # 编译模型（Focal Loss + 类别权重）
        class_weights = [0.5, 10.0, 1.0, 1.0, 1.0]  # 给A类（1）加权重
        focal_loss = SparseFocalLoss(alpha=class_weights, gamma=2.0)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=focal_loss,
            metrics=['accuracy']
        )
        model.summary()

        # 训练模型
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        print("\n开始训练...")
        history = model.fit(
            X_train, y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint, tensorboard]
        )

        # 保存最后一轮模型
        model.save(model_path)
        print(f"模型保存完成: {model_path}")
        print(f"最优模型保存完成: {best_model_path}")

        # 绘制训练曲线
        plot_history_tf(history)

    # 5. 预测并绘制混淆矩阵
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    plot_heat_map(y_test, y_pred)


if __name__ == '__main__':
    main()