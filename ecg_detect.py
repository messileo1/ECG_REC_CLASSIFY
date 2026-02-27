import numpy as np
from matplotlib import pyplot as plt


class ECGDetect:
    """
    ECG QPS、T、P检测
    """

    def __init__(self, signal):
        """
        :param signal: ecg信号
        """
        self.signal = signal
        # 小波变换level
        self.level = 4
        # 信号长度
        self.length = len(signal)
        # 是否绘制波形
        self.plt_flag = False

    def __biorthogonal_wavelets__(self):
        """
        对输入信号进行二进样条4层小波变换
        :return:
        """
        print('biorthogonal wavelets')
        # 存储概貌信息
        swa = np.zeros((self.level, self.length))
        # 存储细节信息
        swd = np.zeros((self.level, self.length))

        # 低通滤波器 1/4 3/4 3/4 1/4
        # 高通滤波器 -1/4 -3/4 3/4 1/4
        # 二进样条小波
        for i in range(self.length - 3):
            swa[0, i + 3] = (1 / 4 * self.signal[i + 3]
                             + 3 / 4 * self.signal[i + 2]
                             + 3 / 4 * self.signal[i + 1]
                             + 1 / 4 * self.signal[i])
            swd[0, i + 3] = (-1 / 4 * self.signal[i + 3]
                             - 3 / 4 * self.signal[i + 2]
                             + 3 / 4 * self.signal[i + 1]
                             + 1 / 4 * self.signal[i])

        j = 1
        while j < self.level:
            for i in range(self.length - 24):
                swa[j, i + 24] = (1 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 0]
                                  + 3 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 1]
                                  + 3 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 2]
                                  + 1 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 3])
                swd[j, i + 24] = (-1 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 0]
                                  - 3 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 1]
                                  + 3 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 2]
                                  + 1 / 4 * swa[j - 1, i + 24 - (2 << (j - 1)) * 3])
            j += 1

        if self.plt_flag:
            # 画出原信号和小波变化近似系数、细节系数
            plt.figure(figsize=(16, 10))
            # 调整子图之间的垂直间距
            plt.subplots_adjust(hspace=0.5)

            plt.subplot(self.level + 1, 1, 1)
            plt.plot(self.signal)
            plt.grid()
            plt.axis('tight')
            plt.title(
                'ECG signal along with its approximation coefficients and detail coefficients at scales j=1,2,3,4')

            for i in range(self.level):
                plt.subplot(self.level + 1, 2, 2 * i + 3)
                plt.plot(swa[i, :])
                plt.axis('tight')
                plt.grid()
                plt.xlabel('time')
                plt.ylabel('a ' + str(i + 1))

                plt.subplot(self.level + 1, 2, 2 * i + 4)
                plt.plot(swd[i, :])
                plt.axis('tight')
                plt.grid()
                plt.ylabel('d ' + str(i + 1))
            plt.show()
        print('\t swa shape {}'.format(swa.shape))
        print('\t swd shape {}'.format(swd.shape))
        return swa, swd

    def __find_extrema__(self, swd):
        """
        获取正负极大值
        :param swd: 二进样条小波变换细节系数
        :return: 返回各尺度小细节系数的极大值、极小值位置
                如果当前采样点不是极值点，将其设置为0
        """
        print('find extrema')
        # 创建一个与示例数组形状相同的全零数组
        ddw = np.zeros_like(swd, dtype=np.int32)
        pddw = np.copy(ddw)
        nddw = np.copy(ddw)

        # swd中大于0的元素为True，否则为False。然后，将这个布尔数组与swd相乘，由于布尔数组会被自动转换为0和1，所以相乘的结果就是将小于等于0的元素置为0
        posw = swd * (swd > 0)

        # 斜率大于0
        pdw = ((posw[:, :self.length - 1] - posw[:, 1:self.length]) < 0).astype(int)

        # 正极大值点
        pddw[:, 1:self.length - 1] = (pdw[:, :self.length - 2] - pdw[:, 1:self.length - 1]) > 0

        # 细节系数小于0的点
        negw = swd * (swd < 0)

        # 斜率小于0
        ndw = ((negw[:, :self.length - 1] - negw[:, 1:self.length]) > 0).astype(int)

        # 负极大值点
        nddw[:, 1:self.length - 1] = (ndw[:, :self.length - 2] - ndw[:, 1:self.length - 1]) > 0

        # 或运算
        ddw = pddw | nddw
        ddw[:, 0] = 1
        ddw[:, self.length - 1] = 1

        # 求出极值点的值，其它点置0
        wpeak = ddw * swd
        wpeak[:, 0] += 0
        wpeak[:, self.length - 1] += 0

        if self.plt_flag:
            # 画出原信号和各尺度下的极值点
            plt.figure(figsize=(16, 10))
            # 调整子图之间的垂直间距
            plt.subplots_adjust(hspace=0.5)

            plt.subplot(self.level + 1, 1, 1)
            plt.plot(self.signal)
            plt.grid()
            plt.axis('tight')
            plt.title('ECG signal extreme points of detail coefficients at scales j=1,2,3,4')

            for i in range(self.level):
                plt.subplot(self.level + 1, 1, i + 2)
                # 细节系数波形图
                plt.plot(swd[i, :], color='blue', linewidth=1)
                # 极值波形图
                plt.plot(wpeak[i, :], color='red', linewidth=0.5)
                plt.axis('tight')
                plt.grid()
                plt.xlabel('time')
                plt.ylabel('d ' + str(i + 1) + ' extrema')
            plt.show()

        print('\t wpeak shape {}'.format(wpeak.shape))
        return wpeak

    def __filter_r__(self, coefficients):
        """
        我们能够看出R波的值要明显大于其它位置的值，这样我们就能够设置一个可靠的阈值（将全部点分为4部分。求出每部分最大值的平均值T，阈值为T/3）来提取一组相邻的最大最小值对；这样最大最小值间的过0点就是相应于原始信号的R波点
        :param coefficients: 二进样条小波变换某种尺度下细节系数
        :return: interva, thnega, thposi
            interva: 如果采样点不满足阈值过滤条件，将其设置为0
            thnega: 正极大值的平均
            thposi: 负极大值的平均
        """
        print('filter R wave')
        posi = coefficients * (coefficients > 0)
        nega = coefficients * (coefficients < 0)
        part_len = round(self.length / 4)
        # 求正极大值的平均
        thposi = (np.max(posi[:part_len])
                  + np.max(posi[part_len:2 * part_len])
                  + np.max(posi[2 * part_len:3 * part_len])
                  + np.max(posi[3 * part_len:4 * part_len])) / 4
        # 筛选R波
        posi = posi > (thposi / 3)
        # 求负极大值的平均
        thnega = (np.min(nega[:part_len])
                  + np.min(nega[part_len:2 * part_len])
                  + np.min(nega[2 * part_len:3 * part_len])
                  + np.min(nega[3 * part_len:4 * part_len])) / 4
        # 筛选R波
        nega = -1 * (nega < (thnega / 4))
        # 找到数组sum中非零元素的索引
        sum = posi + nega
        loca = np.where(sum)[0]
        print('\t loca {}'.format(loca))
        # 计算数组loca中相邻元素之间的差异，并将结果存储在数组diff中
        diff = np.zeros(len(loca) - 1)
        for i in range(len(loca) - 1):
            # 如果相邻元素之间的距离小于80，说明它们足够接近，可以计算差值，否则将差值设为0
            if abs(loca[i] - loca[i + 1]) < 80:
                diff[i] = sum[loca[i]] - sum[loca[i + 1]]
            else:
                diff[i] = 0
        # 找到极值对的索引
        loca2 = np.where(diff == -2)[0]
        print('\t loca2 {}'.format(loca2))
        # 负极大值点
        interva = np.zeros(len(sum))
        interva[loca[loca2]] = sum[loca[loca2]]
        # 正极大值点
        interva[loca[loca2 + 1]] = sum[loca[loca2 + 1]]
        if self.plt_flag:
            # 画出原信号和某种尺度下细节系数极值波形图
            plt.figure(figsize=(16, 10))
            # 调整子图之间的垂直间距
            plt.subplots_adjust(hspace=0.5)

            plt.subplot(2, 1, 1)
            plt.title('ECG signal')
            plt.plot(self.signal)

            plt.subplot(212)
            plt.title('coefficients and interva')
            # 某种尺度下细节系数极值波形图
            plt.plot(coefficients, color='blue', linewidth=1)
            # 处理后的极值波形图
            plt.plot(interva, color='red', linewidth=0.5)
            plt.show()

        print('\t interva shape {}'.format(interva.shape))
        print('\t thnega = {}'.format(thnega))
        print('\t thposi = {}'.format(thposi))
        return interva, thnega, thposi

    def __compensation__(self, r_loca, thnega, thposi, wpeak):
        """
        删除多检点，补偿漏检点
        :param r_loca: 长度和采样点一样，如果采样点不是R波峰值点，其值为0
        :param thnega: 正极大值的平均
        :param thposi: 负极大值的平均
        :param wpeak: 各尺度小细节系数的极大值、极小值位置
        :return:
        """
        print('compensation')
        num = 1
        while num != 0:
            num = 0
            # 查找非0点所在索引
            r_index = np.where(r_loca)[0]
            # 计算相邻R点距离
            R_R = r_index[1:] - r_index[:-1]
            # 计算平均距离
            RR_mean = np.mean(R_R)
            for i in range(1, len(r_index)):
                # 错检
                if (r_index[i] - r_index[i - 1]) <= 0.4 * RR_mean:
                    num += 1
                    if self.signal[r_index[i]] > self.signal[r_index[i - 1]]:
                        r_loca[r_index[i - 1]] = 0
                    else:
                        r_loca[r_index[i]] = 0
        num = 2
        while num > 0:
            num -= 1
            # 查找非0点所在索引
            r_index = np.where(r_loca)[0]
            # 计算相邻R点距离
            R_R = r_index[1:] - r_index[:-1]
            # 计算平均距离
            RR_mean = np.mean(R_R)
            for i in range(1, len(r_index)):
                # 漏检
                if (r_index[i] - r_index[i - 1]) > 1.6 * RR_mean:
                    Mj_adjust = wpeak[4, r_index[i - 1] + 80:r_index[i] - 80]
                    points1 = (r_index[i] - 80) - (r_index[i - 1] + 80) + 1
                    adjust_posi = np.where(Mj_adjust > 0, Mj_adjust, 0)
                    adjust_posi = np.where(adjust_posi > thposi / 4, 1, 0)
                    adjust_nega = np.where(Mj_adjust < 0, Mj_adjust, 0)
                    adjust_nega = np.where(adjust_nega < -thnega / 5, -1, 0)
                    interva = adjust_posi + adjust_nega
                    loca = np.where(interva != 0)[0]
                    diff = interva[loca[:-1]] - interva[loca[1:]]
                    loca2 = np.where(diff == -2)[0]
                    interva2 = np.zeros(points1)
                    for j in range(len(loca2)):
                        interva2[loca[loca2[j]]] = interva[loca[loca2[j]]]
                        interva2[loca[loca2[j] + 1]] = interva[loca[loca2[j] + 1]]
                    j = 0
                    while j < points1:
                        if interva2[j] == -1:
                            mark1 = j
                            j += 1
                            while j < points1 and interva2[j] == 0:
                                j += 1
                            mark2 = j
                            mark3 = round((abs(Mj_adjust[mark2]) * mark1 + mark2 * abs(Mj_adjust[mark1])) /
                                          (abs(Mj_adjust[mark2]) + abs(Mj_adjust[mark1])))
                            r_loca[r_index[i - 1] + 80 + mark3 - 10] = 1
                            j += 60
                        j += 1
        # 查找非0点所在索引
        r_index = np.where(r_loca)[0]
        # 计算相邻R点距离
        R_R = r_index[1:] - r_index[:-1]
        # 计算平均距离
        RR_mean = np.mean(R_R)
        print('\t RR_mean = ', RR_mean)
        return RR_mean

    def __plt_qrs__(self, q_loca, r_loca, s_loca):
        """
        绘制QRS波形图
        :param q_loca: QRS波起点
        :param r_loca: QRS中R点位置
        :param s_loca: QRS波终点
        :return:
        """
        # 画出原信号和QRS所在位置波形图
        plt.figure(figsize=(16, 6))
        plt.title('ECG signal and QRS')

        # ECG信息
        plt.plot(self.signal)
        # Q点
        plt.plot(q_loca, color='black', linewidth=0.5)
        # R点
        plt.plot(r_loca, color='red', linewidth=0.5)
        # S点
        plt.plot(s_loca, color='black', linewidth=0.5)
        plt.show()

    def find_qrs(self):
        """
        查找QRS波
        :return:
        """
        # 1. 二进样条小波变换
        swa, swd = self.__biorthogonal_wavelets__()

        # 2. 获取正负极大值
        wpeak = self.__find_extrema__(swd)

        # 3. 设置阈值，筛选R波
        Mj3 = wpeak[2, :]
        interva, thnega, thposi = self.__filter_r__(Mj3)

        # 4. 求正负极值对过零，即R波峰值，并检测出QRS波起点及终点
        Mj1 = wpeak[0, :]
        q_loca = np.zeros(self.length)  # QRS起点
        r_loca = np.zeros(self.length)  # R波波峰
        s_loca = np.zeros(self.length)  # QRS终点
        # 保存R点所在的索引
        r_index = np.zeros(self.length)
        # R点个数
        r_count = 0
        i = 0
        j = 0
        while i < self.length:
            # 如果是负极值点
            if interva[i] == -1:
                # 标记负极值点
                mark1 = i
                i = i + 1
                # 查找正极值点
                while i < self.length and interva[i] == 0:
                    i = i + 1
                # 标记正极值点
                mark2 = i

                # 求极大值对的过零点 已知两个点（mark1,Mj3[mark1]）、（mark2,Mj3[mark2]），计算y=0时的x点坐标  (x2y1 - x1y2)/(y2-y1)
                mark_r = round(
                    (abs(Mj3[mark2]) * mark1 + mark2 * abs(Mj3[mark1])) / (abs(Mj3[mark2]) + abs(Mj3[mark1])))
                # 为何 - 10？经验值吧
                mark_r = mark_r - 10
                # R波位置
                r_index[j] = mark_r
                r_loca[mark_r] = 1

                # 求出QRS波起点
                kqs = mark_r
                mark_q = 0
                while kqs > 1 and mark_q < 3:
                    if Mj1[kqs] != 0:
                        mark_q = mark_q + 1
                    kqs = kqs - 1
                q_loca[kqs] = -1

                # 求出QRS波终点
                kqs = mark_r
                mark_s = 0
                while kqs < self.length and mark_s < 3:
                    if Mj1[kqs] != 0:
                        mark_s = mark_s + 1
                    kqs = kqs + 1
                s_loca[kqs] = -1

                i = i + 60
                j = j + 1
                r_count = r_count + 1
            i = i + 1

        # 5.删除多检点，补偿漏检点
        RR_mean = self.__compensation__(r_loca, thnega, thposi, wpeak)
        heart_beat = 60 * 360 / RR_mean
        print('heart beat ', heart_beat)

        # 绘制QRS波形
        # self.__plt_qrs__(q_loca, r_loca, s_loca)
        return heart_beat, q_loca, r_loca, s_loca

    def __find_p__(self, Mj4, q_index, r_index, r_loca):
        """
        P波检测
        :param Mj4: 尺度4下小细节系数的极大值、极小值位置
        :param q_index: QPS波起点所在的索引
        :param r_index: QRS波R点所在的索引
        :param r_loca: 长度和采样点一样，如果采样点不是R波峰值点，其值为0
        :return:
        """
        p_loca = np.zeros_like(r_loca)
        p_loca_begin = np.zeros_like(r_loca)
        p_loca_end = np.zeros_like(r_loca)
        window_size = 100
        for i in range(1, len(r_index)):
            flag = 0
            mark_p = 0
            # 计算相邻R之间的距离
            R_R = r_index[i] - r_index[i - 1]
            for j in range(0, R_R * 2 // 3, 5):
                window_end = q_index[i] - j
                window_begin = window_end - window_size
                if window_begin < r_index[i - 1] + R_R / 3:
                    break
                window_max = np.max(Mj4[window_begin:window_end])
                window_min = np.min(Mj4[window_begin:window_end])
                max_index = np.argmax(Mj4[window_begin:window_end])
                min_index = np.argmin(Mj4[window_begin:window_end])
                if min_index < max_index and (
                        (max_index - min_index) < window_size * 2 / 3) and window_max > 0.01 and window_min < -0.1:
                    flag = 1
                    mark_p = round((max_index + min_index) / 2 + window_begin)
                    p_loca[mark_p - 20] = 1
                    p_loca_begin[window_begin + min_index - 20] = -1
                    p_loca_end[window_begin + max_index - 20] = -1
                    break
            if mark_p == 0 and flag == 0:
                mark_p = round(r_index[i] - R_R / 3)
                p_loca[mark_p - 20] = -1
        return p_loca, p_loca_begin, p_loca_end

    def __plt_p__(self, p_loca, p_loca_begin, p_loca_end):
        """
        绘制P波形图
        :param p_loca: P波极值点
        :param p_loca_begin: P波起点
        :param p_loca_end: P波终点
        :return:
        """
        # 画出原信号和QRS所在位置波形图
        plt.figure(figsize=(16, 6))
        plt.title('ECG signal and P')

        # ECG信息
        plt.plot(self.signal)
        plt.plot(p_loca_begin, color='black', linewidth=0.5)
        plt.plot(p_loca, color='red', linewidth=0.5)
        plt.plot(p_loca_end, color='black', linewidth=0.5)
        plt.show()

    def __find_t__(self, Mj4, r_index, r_loca, s_index):
        """
        T波检测
        :param Mj4: 尺度4下小细节系数的极大值、极小值位置
        :param r_index: QRS波R点所在的索引
        :param r_loca: 长度和采样点一样，如果采样点不是R波峰值点，其值为0
        :param s_index: QPS波终点所在的索引
        :return:
        """
        t_loca = np.zeros_like(r_loca)
        t_loca_begin = np.zeros_like(r_loca)
        t_loca_end = np.zeros_like(r_loca)
        window_size = 100
        for i in range(len(r_index) - 1):
            mark_t = 0
            R_R = r_index[i + 1] - r_index[i]
            for j in range(0, R_R * 2 // 3, 5):
                window_begin = s_index[i] + j
                window_end = window_begin + window_size
                if window_end > r_index[i + 1] - R_R / 4:
                    break
                window_max = np.max(Mj4[window_begin:window_end])
                window_min = np.min(Mj4[window_begin:window_end])
                max_index = np.argmax(Mj4[window_begin:window_end])
                min_index = np.argmin(Mj4[window_begin:window_end])
                if min_index < max_index and (
                        (max_index - min_index) < window_size) and window_max > 0.1 and window_min < -0.1:
                    mark_t = round((max_index + min_index) / 2 + window_begin)
                    t_loca[mark_t - 20] = 1
                    t_loca_begin[window_begin + min_index - 20] = -1
                    t_loca_end[window_begin + max_index - 20] = -1
                    break
            if mark_t == 0:
                mark_t = round(r_index[i] + R_R / 3)
                t_loca[mark_t] = -2
        return t_loca, t_loca_begin, t_loca_end

    def __plt_t__(self, t_loca, t_loca_begin, t_loca_end):
        """
        绘制T波形图
        :param t_loca: P波极值点
        :param t_loca_begin: P波起点
        :param t_loca_end: P波终点
        :return:
        """
        # 画出原信号和QRS所在位置波形图
        plt.figure(figsize=(16, 6))
        plt.title('ECG signal and T')

        # ECG信息
        plt.plot(self.signal)
        plt.plot(t_loca_begin, color='black', linewidth=0.5)
        plt.plot(t_loca, color='red', linewidth=0.5)
        plt.plot(t_loca_end, color='black', linewidth=0.5)
        plt.show()

    def find_pt(self):
        """
        查找P、T波
        :return:
        """
        # 1. 二进样条小波变换
        swa, swd = self.__biorthogonal_wavelets__()

        # 2. 获取正负极大值
        wpeak = self.__find_extrema__(swd)

        # 3. 设置阈值，筛选R波
        Mj4 = wpeak[3, :]
        interva, thnega, thposi = self.__filter_r__(Mj4)

        # 4. 求正负极值对过零，即R波峰值，并检测出QRS波起点及终点
        q_loca = np.zeros(self.length)
        r_loca = np.zeros(self.length)
        s_loca = np.zeros(self.length)
        flag = 0
        i = 0

        while i < self.length:
            # 如果是负极值点
            if interva[i] == -1:
                # 标记负极值点
                mark1 = i
                i += 1
                # 查找正极值点
                while i < self.length and interva[i] == 0:
                    i += 1
                # 标记正极值点
                mark2 = i

                # 求极大值对的过零点 已知两个点（mark1,Mj3[mark1]）、（mark2,Mj3[mark2]），计算y=0时的x点坐标  (x2y1 - x1y2)/(y2-y1)
                mark_r = round(
                    (abs(Mj4[mark2]) * mark1 + mark2 * abs(Mj4[mark1])) / (abs(Mj4[mark2]) + abs(Mj4[mark1])))
                r_loca[mark_r] = 1
                q_loca[mark1] = -1
                s_loca[mark2] = -1
                flag = 1

            if flag == 1:
                i += 200
                flag = 0
            else:
                i += 1

        # 5.删除多检点，补偿漏检点
        self.__compensation__(r_loca, thnega, thposi, wpeak)

        # 绘制QRS波形
        # self.__plt_qrs__(q_loca, r_loca, s_loca)

        # 获取QRS波R点所在的索引
        r_index = np.where(r_loca)[0]
        # 获取QPS波起点所在的索引
        q_index = np.where(q_loca)[0]
        # 获取QPS波终点所在的索引
        s_index = np.where(s_loca)[0]

        # 5. P波检测
        p_loca, p_loca_begin, p_loca_end = self.__find_p__(Mj4, q_index, r_index, r_loca)

        # 绘制P波形
        #self.__plt_p__(p_loca, p_loca_begin, p_loca_end)

        # 6. T波检测
        t_loca, t_loca_begin, t_loca_end = self.__find_t__(Mj4, r_index, r_loca, s_index)

        # 绘制T波形
        #self.__plt_t__(t_loca, t_loca_begin, t_loca_end)
        return p_loca, p_loca_begin, p_loca_end, t_loca, t_loca_begin, t_loca_end
