# coding: utf-8

import warnings

warnings.filterwarnings('ignore')

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import *
# from tkinter import messagebox
import math
import pandas as pd
import numpy as np
from tda import TDA
# from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # NavigationToolbar2TkAgg
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
import matplotlib.font_manager as fm

matplotlib.rc('font', family='Arial Unicode MS', )  # 解决图上文字乱码的问题

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import fire
from sklearn import metrics
# from matplotlib import pyplot
import time
from sklearn import metrics
import torch
from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, BatchNormalization, Activation
from keras import backend as K


class Interface(Tk):
    def __init__(self):
        super().__init__()  # super代表父类的定义，而不是父类的对象
        self.title("OCS-TGBM")
        self.geometry('1200x850+30+30')
        self.resizable(0, 1)
        # self.configure(bg='white')
        # messagebox.showinfo('Welcome',
        #                     '\t\tWelcome to OCS-TGBM！！\nAn Intelligent Analysis System of Organic Chemical Synthesis '
        #                     'Based on TDA and LightGBM')
        self.create_widget()
        # 调用mainloop()方法，进入事件循环
        self.mainloop()

    def create_widget(self):
        self.frame1 = Frame(self)
        self.frame1.place(relx=0.03, rely=0.1, relwidth=0.27, relheight=0.8)

        self.frame2 = Frame(self, bg='white')
        self.frame2.place(relx=0.35, rely=0, relwidth=0.65, relheight=1)

        self.frame3 = Frame(self.frame2, bg='white')
        self.frame3.place(relx=0.005, rely=0, relwidth=0.49, relheight=0.49)

        self.frame4 = Frame(self.frame2, bg='white')
        self.frame4.place(relx=0.505, rely=0, relwidth=0.49, relheight=0.49)

        self.frame5 = Frame(self.frame2, bg='white')
        self.frame5.place(relx=0, rely=0.51, relwidth=0.45, relheight=0.48)

        self.frame6 = Frame(self.frame2, bg='white')
        self.frame6.place(relx=0.45, rely=0.5, relwidth=0.55, relheight=0.49)

        self.label1 = Label(self, text=' Welcome to OCS-TGBM！', fg='black',
                            font=('Microsoft YaHei', 23, 'bold', 'italic'), anchor='center')
        self.label1.pack(anchor='nw')

        self.label2 = Label(self, text=' OCS-TGBM: Intelligent Analysis of Organic Chemical Synthesis\n'
                                       ' Based on Topological Data Analysis and LightGBM', fg='black',
                            font=('Microsoft YaHei', 10, 'bold'), justify='left', anchor='center')
        self.label2.pack(anchor='nw')

        self.label3 = Label(self.frame1, text='Select Data Analyzed:', fg='black', font=('bold', 8))
        self.label3.place(relx=0.02, rely=0.12, relwidth=0.4, relheight=0.05)

        options1 = ['Yield Rate', 'Feature Descriptor', 'Interactive Feature', 'Prediction Data']
        self.var1 = StringVar()
        self.var1.set(options1[0])
        self.menu1 = OptionMenu(self.frame1, self.var1, *options1)
        self.menu1.place(relx=0.02, rely=0.17, relwidth=0.45, relheight=0.05)
        self.menu1.config(bg='white', font=('bold', 10))

        self.btn1 = Button(self.frame1, font=('bold', 10), bg="white")
        self.btn1["text"] = 'Upload'
        self.btn1['command'] = self.uploadfun
        self.btn1.place(relx=0.55, rely=0.17, relwidth=0.45, relheight=0.048)

        self.btn2 = Button(self.frame1, text='TDA Clustering', font=('bold', 10), bg="white", command=self.tda_cluster)
        self.btn2.place(relx=0.09, rely=0.27, relwidth=0.82, relheight=0.05)

        self.btn3 = Button(self.frame1, text='Interactive Analysis', font=('bold', 10), bg="white",
                           command=self.interation_analysis)
        self.btn3.place(relx=0.09, rely=0.37, relwidth=0.82, relheight=0.05)

        self.label4 = Label(self.frame1, text='Enter Sampling Loops:', fg='black', font=('bold', 8))
        self.label4.place(relx=0.02, rely=0.47, relwidth=0.4, relheight=0.05)

        self.var2 = IntVar(self)
        self.var2.set(6)
        self.entry1 = Entry(self.frame1, textvariable=self.var2, font=('bold', 10))
        self.entry1.place(relx=0.08, rely=0.52, relwidth=0.3, relheight=0.05)

        self.label5 = Label(self.frame1, text='Decide Sampling Number In A Loop:', fg='black', font=('bold', 8))
        self.label5.place(relx=0.44, rely=0.47, relwidth=0.54, relheight=0.05)

        self.var3 = IntVar(self)
        self.var3.set(395)
        self.entry2 = Entry(self.frame1, textvariable=self.var3, font=('bold', 10))
        self.entry2.place(relx=0.505, rely=0.52, relwidth=0.42, relheight=0.05)

        self.btn4 = Button(self.frame1, text='Diversity Sampling', font=('bold', 10), bg="white",
                           command=self.diversity_sample)
        self.btn4.place(relx=0.09, rely=0.62, relwidth=0.82, relheight=0.05)

        self.btn5 = Button(self.frame1, text='Intelligent Prediction', font=('bold', 10), bg="white",
                           command=self.yield_prediction)
        self.btn5.place(relx=0.09, rely=0.72, relwidth=0.82, relheight=0.05)

        self.download_btn1 = Button(self.frame1, text='Get Sampling Data', font=('bold', 10), bg="white",
                                    command=self.get_sampling_data)
        self.download_btn1.place(relx=0.02, rely=0.82, relwidth=0.45, relheight=0.05)

        self.download_btn2 = Button(self.frame1, text='Get Prediction Results', font=('bold', 10), bg="white",
                                    command=self.get_prediction_results)
        self.download_btn2.place(relx=0.55, rely=0.82, relwidth=0.45, relheight=0.05)

        self.quit = Button(self.frame1, text='Quit', font=('bold', 10), bg="white", command=self.destroy)
        self.quit.place(relx=0.4, rely=0.92, relwidth=0.2, relheight=0.05)

    def uploadfun(self):
        global asistant_var
        asistant_var = 0
        file = askopenfilename(title='Choose Uploaded File', initialdir='c:', filetypes=[('CSV file', '.csv')])
        if self.var1.get() == 'Yield Rate' and file != '':
            global y
            y = np.loadtxt(file, delimiter=",", skiprows=1)
        elif self.var1.get() == 'Feature Descriptor' and file != '':
            global data1
            data1 = np.loadtxt(file, delimiter=",", skiprows=1)
        elif self.var1.get() == 'Interactive Feature' and file != '':
            global data2
            data2 = pd.read_csv(file, header=0)
        elif self.var1.get() == 'Prediction Data' and file != '':
            asistant_var = 1
            global predicted_data
            predicted_data = pd.read_csv(file, header=0)

    # TDA聚类

    def tda_cluster(self):

        def distance(a, b):
            s = sum((k - j) ** 2 for k, j in zip(a, b))
            return math.sqrt(s)

        def l_inf(n, i):
            L = 0
            for j in range(len(n)):
                dist = distance(n[i], n[j])
                if dist > L:
                    L = dist
            return L

        t = TDA(distance, [(l_inf, 15, 0.6)], 5)
        t.fit(data1)
        # t.binums, len(t.clusters)
        image1 = t.dye(lambda d, i: y[i], figsizex=6, figsizey=4, Pk=1, title='TDA Cluster', DPI=80)
        # plt.savefig('tda.tif', dpi=300, bbox_inches='tight')
        # plt.show()
        image1.set_facecolor('white')
        self.canvas1 = FigureCanvasTkAgg(image1, self.frame3)
        self.canvas1.draw()  # 以前的版本使用show方法，matplotlib 2.2之后不再推荐show, 用draw代替，但是用show不会报错，会显示警告
        toolbar1 = NavigationToolbar2Tk(self.canvas1, self.frame3, pack_toolbar=False)
        # matplotlib 2.2版本之后推荐使用NavigationToolbar2Tk，若使用NavigationToolbar2TkAgg会警告
        toolbar1.update()
        toolbar1.pack(side='top', fill='both')
        self.canvas1.get_tk_widget().pack(fill='both', expand=1)

    # # 多因素方差分析
    # (Interaction-based association analysis between reaction conditions and yield：反应条件与产率的交互作用分析）

    def interation_analysis(self):
        df = pd.DataFrame(data2)
        # print(df.head())
        df1 = pd.DataFrame()
        data_list = []
        for i in df.Ligand.unique():
            for j in df.Aryl.unique():
                data = df[(df.Ligand == i) & (df.Aryl == j)]['y'].values
                data_list.append(data)
                df1 = df1.append(pd.DataFrame(data, columns=pd.MultiIndex.from_arrays([[i], [j]])).T)

        df1 = df1.T
        # 查看各组数量分布
        df1.count().to_frame()
        df_mean = df1.mean().to_frame().unstack().round(1)
        df_mean.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        df_mean = df_mean[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']]

        # print(df_mean)

        # 定义一个绘图函数
        def draw_pics(data, feature):
            image2 = plt.figure(figsize=(6, 4), dpi=95)
            for i in data.index:
                plt.plot(data.columns, data.loc[i,], label=i, marker='*', linewidth=0.8, linestyle='-.',
                         markersize=4.5)

            plt.legend(edgecolor='black', title='Ligand', fontsize=8)
            plt.title("Interaction Analysis", fontsize=9)
            plt.xlabel(feature, fontdict={'fontsize': 8.5})
            plt.ylabel("Estimated Marginal Mean", fontdict={'fontsize': 8.5})
            # plt.show()
            return image2

        image2 = draw_pics(df_mean, 'Aryl')
        image2.set_facecolor('white')
        self.canvas2 = FigureCanvasTkAgg(image2, self.frame4)
        self.canvas2.draw()  # 以前的版本使用show方法，matplotlib 2.2之后不再推荐show, 用draw代替，但是用show不会报错，会显示警告
        toolbar2 = NavigationToolbar2Tk(self.canvas2, self.frame4, pack_toolbar=False)
        # matplotlib 2.2版本之后推荐使用NavigationToolbar2Tk，若使用NavigationToolbar2TkAgg会警告
        toolbar2.update()
        toolbar2.pack(side='top', fill='both')
        self.canvas2.get_tk_widget().pack(fill='both', expand=1)

    # 多样性抽样

    # 此函数用于更新Train_x
    # data_var来源于相似度计算，是x_test中的数据

    def upTrain_x(self, data_var, X):
        final_update = torch.tensor([item.detach().numpy() for item in data_var])
        return torch.cat((X, final_update), dim=0)

    # 用于更新train_y
    # # train_y = train_y + (data_var <- test_y)

    def upTrain_y(self, data_var, Y):
        return torch.cat((Y, data_var), dim=0)

    # 用于更新test_y：
    # # test_y =>
    # #          1. test_y
    # #          2. data_var -> train_y
    # # return: (test_y,data_var)

    def upTest_y(self, line_num, Y):
        out = np.array([], dtype='float64')
        data_var = np.array([], dtype='float64')
        for i in range(len(Y)):
            if i not in line_num:
                out = np.append(out, Y[i])
            else:
                data_var = np.append(data_var, Y[i])
        return [torch.tensor(out), torch.tensor(data_var)]

    # 此函数用于更新x
    # 删除x_test中的数据，不做返回，返回在相似度计算时已做
    # # test_x =>
    # #          1. test_x
    # #          2. data_var -> train_x

    def upTest_x(self, line_num, X):
        out = []
        for i in range(len(X)):
            if i not in line_num:
                out.append(X[i])
        return torch.stack(out, dim=1).T  # 转化为torch，注意需要转至

    # 此函数用于计算训练集与测试集相似度
    # # x_train: 训练集的x
    # # x_test: 测试集的x
    # # return:
    # # #       1. line_num
    # # #       2. data_var <- x_test

    def CalSim(self, x_train, x_test, n):
        similarity_list = []
        # line_num,data_var=[],[]
        for i in range(len(x_test)):
            s_ = torch.cosine_similarity(x_test[i], x_train, dim=-1)  # 计算未标记数据的与所有标记数据的相似度
            s_max = torch.max(s_).item()  # 返回每一个未标记数据对应的最大相似度
            similarity_list.append(s_max)  # 将最大相似度添加到相似度列表中
        df = pd.DataFrame(zip(list(enumerate(x_test)), similarity_list), columns=['index', 'similarity'])
        df_sorted = df.sort_values(by=['similarity'], ascending=True)  # 将最大相似度升序排列
        df_index = df_sorted['index'].values
        update_list = list(df_index[:n])  # 选择前10名作为候选对象
        # 将update_list排序
        df2 = pd.DataFrame(update_list, columns=["num", "data"])
        df2 = df2.sort_values(by=['num'], ascending=True)  # 升序排列
        line_num = list(df2["num"])
        data_var = list(df2["data"])
        return [line_num, data_var]

    def Upgrade(self, n, m, train_x, train_y, test_x, test_y):
        train_x_ = train_x
        train_y_ = torch.tensor(train_y)
        test_x_ = test_x
        test_y_ = torch.tensor(test_y)
        count = 1
        textvar = "Sampling...........\n\n  After 1 sampling loop, train set includes %4d samples, " \
                  "test set includes %4d samples." % (train_x_.shape[0], train_x_.shape[0])
        self.text1.insert('insert', textvar + '\n\n')
        self.text1.update()
        # print("New train_x = ({},{})\t".format(train_x_.shape[0], train_x_.shape[1]), end="\t")
        # print("New test_x = ({},{})\t".format(test_x_.shape[0], test_x_.shape[1]), end="\n")

        while count < n:
            out_sim = self.CalSim(train_x_, test_x_, m)
            # 更新train_x_
            train_x_ = self.upTrain_x(out_sim[1], train_x_)
            # print("New train_x = ({},{})\t".format(train_x_.shape[0], train_x_.shape[1]), end="\t")
            # # 更新test_x_
            test_x_ = self.upTest_x(out_sim[0], test_x_)
            # print("New test_x = ({},{})\t".format(test_x_.shape[0], test_x_.shape[1]), end="\t")
            # # 更新test_y_
            out_uptest_y = self.upTest_y(out_sim[0], test_y_)
            test_y_ = out_uptest_y[0]
            # print("New test_y_ = ({},)\t".format(test_y_.shape[0]), end="\t")
            # # 更新train_y_
            # print(train_y_)
            train_y_ = self.upTrain_y(out_uptest_y[1], train_y_)
            # print("New train_y_ = ({},)\t".format(train_y_.shape[0]), end="\n")
            textvar = 'Sampling...........\n\n  After %d sampling loops, train set includes %4d samples, ' \
                      'test set includes %4d samples.' % (count + 1, train_x_.shape[0], test_x_.shape[0])
            self.text1.insert('insert', textvar + '\n\n')
            self.text1.update()
            count += 1

        # return (train_x_,train_y_,test_x_,test_y_)                                      # 返回的都是tensor
        return [train_x_.numpy(), train_y_.numpy(), test_x_.numpy(), test_y_.numpy()]  # 将所有的转化为array

    def diversity_sample(self):
        X = torch.from_numpy(data1)

        # 将数据分为训练集和测试集

        x_training, x_unlabeled, y_training, y_unlabeled = train_test_split(X, y,
                                                                            train_size=int(self.entry2.get()) / len(y))

        self.text1 = Text(self.frame5, font=('bold', 10), bg="white")
        self.text1.place(relx=0.1, rely=0, relwidth=0.89, relheight=0.6)

        # 创建垂直方向的 Scrollbar 控件
        self.scrollbar1 = Scrollbar(self.frame5, orient=VERTICAL)
        self.scrollbar1.place(relx=0.99, rely=0, relwidth=0.01, relheight=0.6)
        # 将 Scrollbar 控件与 Text 控件关联
        self.text1.config(yscrollcommand=self.scrollbar1.set)
        self.scrollbar1.config(command=self.text1.yview)

        a = self.Upgrade(int(self.entry1.get()), int(self.entry2.get()), x_training, y_training,
                         x_unlabeled, y_unlabeled)

        textvar = 'Sampling ends...........'
        self.text1.insert('insert', textvar)
        self.text1.update()
        self.text1.config(state=DISABLED)
        self.X_train = a[0]
        self.X_test = a[2]
        self.y_train = a[1]
        self.y_test = a[3]
        # print(X_train)
        # print(X_test)
        # print(y_train)
        # print(y_test)
        # print(a)

    def get_sampling_data(self):
        file = askdirectory(title='Save File', initialdir='c:')
        # print(file)
        if file != '':
            np.savetxt(file + '/feature_descriptor_train.csv', self.X_train, delimiter=',')
            np.savetxt(file + '/feature_descriptor_test.csv', self.X_test, delimiter=',')
            np.savetxt(file + '/yield_rate_train.csv', self.y_train, delimiter=',')
            np.savetxt(file + '/yield_rate_test.csv', self.y_test, delimiter=',')

    # 产率预测

    def yield_prediction(self):
        gbm = lgb.LGBMRegressor(reg_alpha=0.11, reg_lambda=1, min_child_samples=8, min_child_weight=0,
                                colsample_bytree=0.75, subsample=1,
                                num_leaves=51, max_depth=12, min_split_gain=0, learning_rate=0.1, n_estimators=515)
        gbm.fit(self.X_train, self.y_train)
        # 测试机预测
        start1 = time.time()
        self.x_test_y_pred = gbm.predict(self.X_test)
        end1 = time.time()
        # print(x_test_y_pred)
        # print("循环运行时间:%.4f秒" % (end1 - start1))
        # 评估回归性能

        self.text2 = Text(self.frame5, font=('bold', 10), bg="white")
        self.text2.place(relx=0.1, rely=0.63, relwidth=0.89, relheight=0.35)
        textvar = " Model Evaluation Index :\n\n" \
                  '  R2:                                   %.4f\n\n' \
                  '  Root Mean Squared Error:  %.4f\n\n' \
                  "  Mean Absolute Error:         %.4f\n\n" \
                  '  Run Time(s):                     %.4fs\n' \
                  % ((metrics.r2_score(self.y_test, self.x_test_y_pred),
                      np.sqrt(metrics.mean_squared_error(self.y_test, self.x_test_y_pred)),
                      metrics.mean_absolute_error(self.y_test, self.x_test_y_pred),
                      end1 - start1))
        self.text2.insert('insert', textvar)
        self.text2.update()

        # 创建垂直方向的 Scrollbar 控件
        self.scrollbar2 = Scrollbar(self.frame5, orient=VERTICAL)
        self.scrollbar2.place(relx=0.99, rely=0.63, relwidth=0.01, relheight=0.35)
        # 将 Scrollbar 控件与 Text 控件关联
        self.text2.config(yscrollcommand=self.scrollbar2.set)
        self.scrollbar2.config(command=self.text2.yview)
        self.text2.config(state=DISABLED)

        if asistant_var == 1:
            # start2 = time.time()
            self.y_pred = gbm.predict(predicted_data.values[1::, 0::])
            y_pred = np.round(self.y_pred, 4)
            # end2 = time.time()
            # df1 = pd.DataFrame(predicted_data)
            df2 = pd.DataFrame(y_pred, columns=['Yield Prediction'])
            # print(df2)
            df3 = np.array([list(y_pred[0:11]) + ['......'] + list(y_pred[-10:])])

            mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 组件字体
            f = plt.figure(figsize=(4, 10), dpi=88)  # 窗口大小
            f.add_subplot(111, frameon=False, xticks=[], yticks=[])
            # frameon 选择是否显示组建本身的横纵坐标，因为只要显示表格，所以选择False
            # a = df.index[-2] + 1
            # b = df.index[-1] + 1  # 截取数据最后的索引，确定数据数目
            table = plt.table(cellText=df3.T, colLabels=df2.columns, colWidths=[0.07] * df3.shape[1],
                              loc='center', cellLoc='center')
            table.auto_set_font_size(FALSE)
            table.set_fontsize(10)  # 字体大小
            table.scale(6, 2.5)  # 每一个小表格的尺寸
            # plt.show()
            self.canvas3 = FigureCanvasTkAgg(f, self.frame6)
            # 将构建的图表放置在canvas上，该canvas是使用FigureCanvasTkAgg组件构建的，然后设置canvas展示
            self.canvas3.draw()
            self.canvas3.get_tk_widget().place(relx=0, rely=0, relwidth=0.97, relheight=0.97)

            self.scrollbar3 = Scrollbar(self.frame6, orient=VERTICAL)
            self.scrollbar3.place(relx=0.97, rely=0, relwidth=0.03, relheight=0.97)
            self.canvas3.get_tk_widget().config(yscrollcommand=self.scrollbar3.set)
            self.scrollbar3.config(command=self.canvas3.get_tk_widget().yview)

            self.scrollbar4 = Scrollbar(self.frame6, orient=HORIZONTAL)
            self.scrollbar4.place(relx=0, rely=0.97, relwidth=0.97, relheight=0.03)
            self.canvas3.get_tk_widget().config(xscrollcommand=self.scrollbar4.set)
            self.scrollbar4.config(command=self.canvas3.get_tk_widget().xview)

            self.canvas3.get_tk_widget().configure(bg='#FFFFFF', scrollregion=(0, 0, 700, 700))
        else:
            y_test = np.round(self.y_test, 4)
            x_test_y_pred = np.round(self.x_test_y_pred, 4)
            df = pd.concat([pd.DataFrame(y_test, columns=['Observed Yield']),
                            pd.DataFrame(x_test_y_pred, columns=['Predicted Yield'])], axis=1)
            # print(df)
            # print(y_test[0:10])
            df3 = np.array([list(y_test[0:10]) + ['......'] + list(y_test[-10:]),
                            list(x_test_y_pred[0:10]) + ['......'] + list(x_test_y_pred[-10:])])
            # print(df3)

            mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 组件字体
            f = plt.figure(figsize=(5, 10), dpi=88)
            f.add_subplot(111, frameon=False, xticks=[], yticks=[])
            # frameon 选择是否显示组建本身的横纵坐标，因为只要显示表格，所以选择False
            # a = df.index[-2] + 1
            # b = df.index[-1] + 1  # 截取数据最后的索引，确定数据数目
            table = plt.table(cellText=df3.T, colLabels=df.columns, colWidths=[0.07] * df3.shape[1],
                              loc='center', cellLoc='center')
            table.auto_set_font_size(FALSE)
            table.set_fontsize(10)  # 字体大小
            table.scale(6, 2.6)  # 每一个小表格的尺寸
            # plt.show()
            self.canvas3 = FigureCanvasTkAgg(f, self.frame6)
            # 将构建的图表放置在canvas上，该canvas是使用FigureCanvasTkAgg组件构建的，然后设置canvas展示
            self.canvas3.draw()
            self.canvas3.get_tk_widget().place(relx=0, rely=0, relwidth=0.97, relheight=0.97)

            self.scrollbar3 = Scrollbar(self.frame6, orient=VERTICAL)
            self.scrollbar3.place(relx=0.97, rely=0, relwidth=0.03, relheight=0.97)
            self.canvas3.get_tk_widget().config(yscrollcommand=self.scrollbar3.set)
            self.scrollbar3.config(command=self.canvas3.get_tk_widget().yview)

            self.scrollbar4 = Scrollbar(self.frame6, orient=HORIZONTAL)
            self.scrollbar4.place(relx=0, rely=0.97, relwidth=0.97, relheight=0.03)
            self.canvas3.get_tk_widget().config(xscrollcommand=self.scrollbar4.set)
            self.scrollbar4.config(command=self.canvas3.get_tk_widget().xview)

            self.canvas3.get_tk_widget().configure(bg='#FFFFFF', scrollregion=(0, 0, 700, 700))

            # self.__options_f = Frame(self.canvas3.get_tk_widget())
            # self.__options_f.bind("<Configure>", self.refresh_scroll)
            # self.canvas3.get_tk_widget().create_window((500, 600), window=self.__options_f, anchor='nw')

    # def refresh_scroll(self, event):
    #     self.canvas3.get_tk_widget().configure(scrollregion=self.canvas3.get_tk_widget().bbox("all"),
    #                                            width=1000, height=1000)

    def get_prediction_results(self):
        file = askdirectory(title='Save File', initialdir='c:')
        if asistant_var == 1 and file != '':
            np.savetxt(file + '/prediction_result.csv', self.y_pred, delimiter=',')
        if file != '':
            np.savetxt(file + '/test_set_prediction_result.csv', self.x_test_y_pred, delimiter=',')


if __name__ == '__main__':
    interface = Interface()
