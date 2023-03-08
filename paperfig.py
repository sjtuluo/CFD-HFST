# -*- coding: utf-8 -*-

'''
Author: Lorces
Visualization for IBM module in HydroFlow-IBM
Analysis of numerical results
'''


import matplotlib.pyplot as plt
import numpy as np
import os


ab_path = os.getcwd().replace('\\', '/')
AP = str(ab_path + '/CFD-HFST')


class VisIBM():
    # 二维虚拟边界计算域可视化
    def ibm_gc_2D(self):
        fluid = np.loadtxt(str(AP + '/Result/ibm_data/fluid.dat'))
        solid = np.loadtxt(str(AP + '/Result/ibm_data/solid.dat'))
        gc = np.loadtxt(str(AP + '/Result/ibm_data/gc.dat'))
        ibn = np.loadtxt(str(AP + '/Result/ibm_data/ibn.dat'))

        ax = plt.axes()
        ax.scatter(fluid[:, 0], fluid[:, 1], c='aqua', s=1, alpha=0.5)
        ax.scatter(solid[:, 0], solid[:, 1], c='darkgrey', s=1)
        ax.scatter(gc[:, 0], gc[:, 1], c='r', s=5, marker='*')
        ax.scatter(ibn[:, 0], ibn[:, 1], c='k', s=2, alpha=0.5)

        plt.show()

    # 三维虚拟边界计算域可视化
    def ibm_gc_3D(self):
        fluid = np.loadtxt(str(AP + '/Result/ibm_data/fluid.dat'))
        solid = np.loadtxt(str(AP + '/Result/ibm_data/solid.dat'))
        gc = np.loadtxt(str(AP + '/Result/ibm_data/gc.dat'))
        ibn = np.loadtxt(str(AP + '/Result/ibm_data/ibn.dat'))

        ax = plt.axes(projection='3d')
        # ax.scatter3D(fluid[:, 0], fluid[:, 1], fluid[:, 2], c='aqua', s=10)
        ax.scatter3D(solid[:, 0], solid[:, 1], solid[:, 2], c='darkgrey', s=1)
        ax.scatter3D(gc[:, 0], gc[:, 1], gc[:, 2], c='r', s=10, marker='*')
        ax.scatter3D(ibn[:, 0], ibn[:, 1], ibn[:, 2], c='b', s=1, alpha=1)

        plt.show()

    # 二维虚拟边界镜像点可视化
    def ibm_image_2D(self):
        gcimage = np.loadtxt(str(AP + '/Result/ibm_data/gcimage.dat'))
        solid = np.loadtxt(str(AP + '/Result/ibm_data/solid.dat'))
        gc = np.loadtxt(str(AP + '/Result/ibm_data/gc.dat'))
        ibn = np.loadtxt(str(AP + '/Result/ibm_data/ibn.dat'))

        ax = plt.axes()
        for i in range(0, len(gcimage), 4):
            ax.fill(gcimage[:, i], gcimage[:, i + 4], facecolor='aqua', alpha=0.1, zorder=0)
        ax.scatter(solid[:.0], solid[:, 1], c='g', s=5, marker='>', zorder=20)
        ax.scatter(gc[:, 0], gc[:, 1], c='r', s=5, marker='*', zorder=10)
        ax.scatter(ibn[:, 0], ibn[:, 1], c='k', s=1, alpha=0.5, zorder=10)

        plt.show()

    # 二维边界权重插值点可视化
    def ibm_wm_2D(self,dr, cir=None):
        points = np.loadtxt(str(AP + '/Result/ibm_data/points.dat'))
        gcimage = np.loadtxt(str(AP + '/Result/ibm_data/gcimage.dat'))
        gc = np.loadtxt(str(AP + '/Result/ibm_data/gc.dat'))
        ibn = np.loadtxt(str(AP + '/Result/ibm_data/ibn.dat'))

        ax = plt.axes()
        circle = plt.Circle((cir[0], cir[1]), cir[2], color='darkgrey', alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(points[:, 0], points[:, 1], c='aqua', s=5)
        ax.scatter(gcimage[:, 0], gcimage[:, 1], c='darkred', s=5)
        ax.scatter(gc[:, 0], gc[:, 1], c='r', s=20, marker='*')
        ax.scatter(ibn[:, 0], ibn[:, 1], c='k', s=1, alpha=0.5)
        for i in range(0, len(gcimage)):
            circles = plt.Circle((gcimage[i, 0], gcimage[i, 1]), dr, color='gold', alpha=0.1)
            ax.add_patch(circles)
            circles.set_zorder(0)

        plt.show()

    # 二维虚拟边界点法向与切向向量可视化
    def ibm_vec(self):
        n = np.loadtxt(str(AP + '/Result/ibm_data/normal.dat'))
        t = np.loadtxt(str(AP + '/Result/ibm_data/tangent.dat'))

        ax = plt.axes(projection='3d')
        ax.quiver(n[:, 0], n[:, 1], n[:, 2], n[:, 3], n[:, 4], n[:, 5])
        ax.quiver(t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5])
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 0.2)
        plt.show()

    # 二维边界力源项影响域可视化
    def ibm_df_2D(self,cir):
        xdf = np.loadtxt(str(AP + '/Result/ibm_data/delta_f.dat'))
        xdb = np.loadtxt(str(AP + '/Result/ibm_data/delta_ib.dat'))

        ax = plt.axes()
        circle = plt.Circle((cir[0], cir[1]), cir[2], color='r', alpha=0.1)
        ax.add_patch(circle)
        ax.scatter(xdf[:, 0], xdf[:, 1], c='aqua', s=5)
        ax.scatter(xdb[:, 0], xdb[:, 1], c='k', s=1, alpha=0.5)

        for i in range(0, len(xdb)):
            circles = plt.Circle((xdb[i, 0], xdb[i, 1]), cir[3], color='gold', alpha=0.1)
            ax.add_patch(circles)
            circles.set_zorder(0)

        plt.show()

    # 虚拟边界计算域可视化（网格形式）
    def ibm_mesh(self,npoint):
        fluid = np.loadtxt(str(AP + '/Result/ibm_data/fluid.dat'))
        solid = np.loadtxt(str(AP + '/Result/ibm_data/solid.dat'))
        gc = np.loadtxt(str(AP + '/Result/ibm_data/gc.dat'))
        ibn = np.loadtxt(str(AP + '/Result/ibm_data/ibn.dat'))
        gci = np.loadtxt(str(AP + '/Result/ibm_data/gcimage.dat'))
        gcp = np.loadtxt(str(AP + '/Result/ibm_data/points.dat'))

        bgx = [0.1,0.1,-0.1,-0.1]
        bgy = [0.1,-0.1,-0.1,0.1]
        ax = plt.axes()

        ax.fill(bgx,bgy,c='aqua',alpha=0.5,zorder=0)
        for i in range(0, len(fluid), npoint+1):
            ax.plot(fluid[i:i+npoint+1, 0], fluid[i:i+npoint+1, 1], linewidth=0.5,c='k',zorder=2)
        for i in range(0, len(solid), npoint):
            ax.fill(solid[i:i+npoint, 0], solid[i:i+npoint, 1], c='gray', zorder=1)
        for i in range(0, len(gc), npoint):
            ax.fill(gc[i:i+npoint, 0], gc[i:i+npoint, 1], c='r',zorder=1)
        for i in range(0, len(gcp), npoint):
            ax.fill(gcp[i:i+npoint, 0], gcp[i:i+npoint, 1], c='y',zorder=1)
        ax.scatter(gci[:, 0], gci[:, 1], c='b', s=1, alpha=0.7, zorder=3)
        ax.plot(ibn[:, 0], ibn[:, 1], c='k', linewidth=2, linestyle='--',zorder=3)
        ax.set_xlim(-0.06,0.06)
        ax.set_ylim(-0.06,0.06)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        filename = str(AP + '/Result/figures/test.png')

        plt.savefig(filename, dpi=200)

    # 二维运动虚拟边界可视化
    def ibm_ani_2D(self):
        fluid = np.loadtxt(str(AP + '/Result/ibm_data/fluid.dat'))
        solid = np.loadtxt(str(AP + '/Result/ibm_data/solid.dat'))
        gc = np.loadtxt(str(AP + '/Result/ibm_data/gc.dat'))
        ibn = np.loadtxt(str(AP + '/Result/ibm_data/ibn.dat'))
        boundary = np.loadtxt(str(AP + '/Result/ibm_data/boundary.dat'))

        xs_t, xg_t, xb_t = [],[],[]

        istart = 0
        for i in range(0, len(solid)):
            if solid[i, 2] > 0:
                iend = i
                xs_t.append(solid[istart:iend, :])
                istart = i + 1

        istart = 0
        for i in range(0, len(gc)):
            if gc[i, 2] > 0:
                iend = i
                xg_t.append(gc[istart:iend, :])
                istart = i + 1

        istart = 0
        for i in range(0, len(ibn)):
            if ibn[i, 2] == 1:
                iend = i
                xb_t.append(ibn[istart:iend, :])
                istart = i + 1

        for i in range(0,len(xb_t)):
            fig = plt.figure(dpi=300)
            ax = plt.axes()
            ax.scatter(fluid[:, 0], fluid[:, 1], c='aqua', s=1, alpha=0.5,zorder=0)
            ax.fill(boundary[:, 0], boundary[:, 1], facecolor='gray',zorder=0)
            ax.fill(xb_t[i][0:4, 0], xb_t[i][0:4, 1], facecolor='lightgray',zorder=1)
            ax.fill(xb_t[i][4:, 0], xb_t[i][4:, 1], facecolor='lightgray', zorder=1)
            ax.scatter(xg_t[i][:, 0], xg_t[i][:, 1], c='r', s=5, marker='*',zorder=1)
            ax.scatter(xs_t[i][:, 0], xs_t[i][:, 1], c='g', s=2,zorder=1)

            ax.set_xlim(5250,6150)
            ax.set_ylim(11900,12600)
            ax.axhline(y=-0.228,c='k')

            filename = str(AP + '/Result/figures/') + str('%03d' % i) + '.png'
            plt.savefig(filename, dpi=100)
            print('Figure of ibm ' + str('%03d' % i) + '.png ' + 'has been saved!')

            plt.close()


class VisResult():
    # 边界层速度剖面u+/y+图
    def u_boundary(self):
        numerical = np.loadtxt(AP + '/Result/uplus.dat')

        yplus1 = np.arange(0,11.63,0.01)
        uplus1 = yplus1
        yplus2 = np.arange(11.63,10000,1)
        uplus2 = np.log(9.8 * yplus2) / 0.41
        yplus = np.concatenate([yplus1, yplus2])
        uplus = np.concatenate([uplus1, uplus2])

        yplus_num = numerical[:, 0]
        uplus_num = numerical[:, 1]

        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes()
        ax.plot(yplus, uplus,label='Experimental')
        ax.scatter(yplus_num,uplus_num,c='r',s=20,label='Numerical')
        ax.set_xscale("log")

        ax.set_title('Velocity of boundary layer',fontsize=18)
        ax.set_xlabel('y+',fontsize=16)
        ax.set_ylabel('u+',fontsize=16)
        ax.legend(frameon = False,fontsize=12)

        plt.show()

    # 圆柱绕流中轴线速度分布图
    def u_mid(self,u_inf,d):
        num = np.loadtxt(AP + '/Result/cylinder_data/U-INS.DAT')
        exp1 = np.loadtxt(AP + '/Result/cylinder_data/umid.dat')
        exp2 = np.loadtxt(AP + '/Result/cylinder_data/Re3900_u_piv.dat')
        pos = np.loadtxt(AP + '/Result/cylinder_data/position.dat')

        u = num[:,1:]
        u_mean = np.mean(u, axis=0) / u_inf
        x = pos[:,1] / d

        plt.figure(figsize=(7, 8))
        ax = plt.axes()
        ax.plot(x, u_mean, c='r',linewidth=2,linestyle='--',label='Numerical',zorder=0)
        ax.scatter(exp1[:, 0], exp1[:, 1], c='k', s=10, label='Experiment 1',zorder=0)
        ax.scatter(exp2[:, 0], exp2[:, 1], c='w', s=15, marker='<', edgecolors='k', label='Experiment 2', zorder=0)
        ax.fill([-0.5, 0.5, 0.5, -0.5], [-0.4, -0.4, 1, 1], facecolor='lightgray',zorder=1)
        ax.set_xlim(-5,20)
        ax.set_ylim(-0.4,1)
        ax.set_xlabel('X / D', fontsize=18, fontdict={'fontproperties': 'Times New Roman'})
        ax.set_ylabel('u / Uinf', fontsize=18, fontdict={'fontproperties': 'Times New Roman'})
        ax.legend(fontsize=16, loc='lower right',prop={'family': 'Times New Roman', 'size': 16})

        plt.xticks(fontproperties='Times New Roman',fontsize=16)
        plt.yticks(fontproperties='Times New Roman',fontsize=16)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        filename = str(AP + '/Result/figures/') + str('test2') + '.png'
        plt.show()
        #plt.savefig(filename, dpi=200)

    # 无量纲压力系数图
    def u_Cp(self):
        exp = np.loadtxt(AP + '/Result/cylinder_data/Cp_exp.dat')  # Experimental data
        num = np.loadtxt(AP + '/Result/cylinder_data/Cp_num.dat')  # Numerical data

        fig = plt.figure(figsize=(8,6),dpi=200)
        ax = plt.axes()
        ax.plot(exp[:, 0], exp[:, 1], c='k', linestyle='-', linewidth=1.5, label='Numerical')
        ax.scatter(num[:, 0], num[:, 1], c='k', s=30, marker='^', label='Experimental')
        ax.set_title('Velocity of boundary layer', fontsize=18, y=1.02, fontproperties='Times New Roman')
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 180)
        ax.set_xlabel('theta', fontsize=16, fontdict={'fontproperties': 'Times New Roman'})
        ax.set_ylabel('Cp', fontsize=16, fontdict={'fontproperties': 'Times New Roman'})
        ax.legend(frameon=False, fontsize=12, loc='lower right', prop={'family': 'Times New Roman', 'size': 16})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xticks(fontproperties='Times New Roman', fontsize=16)
        plt.yticks(fontproperties='Times New Roman', fontsize=16)

        plt.show()

    # 无量纲阻力系数/升力系数图
    def u_CDCL(self):
        num = np.loadtxt(AP + '/Result/ibm_data/CDCL.dat')  # Numerical data

        ax = plt.axes()
        ax.plot(num[:, 0], num[:, 1], c='k', linestyle='-', label='Drag coefficient')
        ax.plot(num[:, 0], num[:, 2], c='k', linestyle='--', label='lift coefficient')
        ax.set_title('Velocity of boundary layer', fontsize=18, y=1.02, fontproperties='Times New Roman')
        ax.set_ylim(-0.5, 2.0)
        ax.set_xlim(200, 1000)
        ax.set_xlabel('t', fontsize=18, fontdict={'fontproperties': 'Times New Roman'})
        ax.set_ylabel('CD/CL', fontsize=18, fontdict={'fontproperties': 'Times New Roman'})
        ax.legend(frameon=False, fontsize=16, loc='lower right', prop={'family': 'Times New Roman', 'size': 16})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()

    # 速度数据数值模拟与实验结果对比图
    def u_result(self):
        data = np.loadtxt(str(AP + '/Result/wave_data/OBS.dat'))
        u1 = np.loadtxt(str(AP + '/Result/wave_data/u1.dat'))
        w1 = np.loadtxt(str(AP + '/Result/wave_data/w1.dat'))
        u1vof = np.loadtxt(str(AP + '/Result/wave_data/u1vof.dat'))
        w1vof = np.loadtxt(str(AP + '/Result/wave_data/w1vof.dat'))

        scale = np.sqrt(2.2344)
        scale2 = np.sqrt(9.8 / 0.228)

        fig, ax = plt.subplots(2, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.3)

        ax[0].plot(data[:, 0] * scale2 - 4.2, data[:, 1] / scale, c='r', linewidth=1.5)
        ax[0].plot(u1vof[:, 0], u1vof[:, 1], c='b', linestyle='--')
        ax[0].scatter(u1[:, 0], u1[:, 1], c='k', s=10)
        ax[0].set(xlim=(0, 30), ylim=(-0.2, 0.3))
        ax[0].set_ylabel('u', fontsize=16, fontdict={'fontproperties': 'Times New Roman'})
        ax[0].text(20, 0.15, "point 1: u", fontsize=16, fontdict={'fontproperties': 'Times New Roman'})

        ax[1].plot(data[:, 0] * scale2 - 4.6, data[:, 2] / scale, c='r', linewidth=1.5)
        ax[1].plot(w1vof[:, 0], w1vof[:, 1], c='b', linestyle='--')
        ax[1].scatter(w1[:, 0], w1[:, 1], c='k', s=10)
        ax[1].set(xlim=(0, 30), ylim=(-0.2, 0.3))
        ax[1].set_ylabel('w', fontsize=16, fontdict={'fontproperties': 'Times New Roman'})
        ax[1].text(20, 0.15, "point 1: w", fontsize=16, fontdict={'fontproperties': 'Times New Roman'})

        plt.show()
        filename = str(AP + '/Result/figures/wave.png')
        plt.savefig(filename, dpi=100)

    # 快速傅里叶变换频谱分析
    def u_fft(self,pos):
        data = np.loadtxt(AP + '/Result/cylinder_data/V-INS.DAT')

        data_gauge = data[:,pos]
        dg_min = np.min(data_gauge) - (np.max(data_gauge)-np.min(data_gauge))/2
        dg_max = np.max(data_gauge) + (np.max(data_gauge)-np.min(data_gauge))/2
        da_min = np.min(data[:,0]) - (np.max(data[:,0])-np.min(data[:,0]))/1000
        da_max = np.max(data[:,0]) + (np.max(data[:,0])-np.min(data[:,0]))/1000

        data_fft = np.fft.fft(data_gauge)
        frequency = np.fft.fftfreq(data_gauge.size, abs(data[1, 0] - data[0, 0]))

        power = np.abs(data_fft)
        pmax = np.max(power)
        pmax_index = np.argmax(power)
        power = power / pmax

        fig, ax = plt.subplots(2, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.5)

        x_major_loc = plt.MultipleLocator(10)
        y_major_loc = plt.MultipleLocator(0.03)

        ax[0].plot(data[:,0],data_gauge, color='r', linewidth=2)
        ax[0].set_ylabel('Velocity', fontsize=20, fontdict={'fontproperties': 'Times New Roman'})
        ax[0].set_xlabel('Time', fontsize=20, fontdict={'fontproperties': 'Times New Roman'})
        ax[0].set_xlim(da_min,da_max)
        ax[0].set_ylim(dg_min,dg_max)
        ax[0].xaxis.set_major_locator(x_major_loc)
        ax[0].yaxis.set_major_locator(y_major_loc)
        for label in ax[0].get_xticklabels() + ax[0].get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(16)

        ax[1].plot(frequency, power,color='darkblue', linewidth=2)
        ax[1].set_xlim(0, frequency[pmax_index]*7)
        ax[1].set_ylim(0, 1)
        ax[1].set_ylabel('Power', fontsize=20, fontdict={'fontproperties': 'Times New Roman'})
        ax[1].set_xlabel('Frequency', fontsize=20, fontdict={'fontproperties': 'Times New Roman'})
        for label in ax[1].get_xticklabels() + ax[1].get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(16)

        f_text = 'Frequency is'+str('{:8.4f}'.format(frequency[pmax_index]))
        T_text = 'Time is' + str('{:8.4f}'.format(1/frequency[pmax_index]))
        ax[1].text(x = frequency[pmax_index] * 1.2, y = 0.85, s=f_text,fontsize=20,
                   c='darkred',fontdict={'fontproperties': 'Times New Roman'})
        ax[1].text(x = frequency[pmax_index] * 1.2, y = 0.65, s=T_text,fontsize=20,
                   c='darkred',fontdict={'fontproperties': 'Times New Roman'})
        plt.show()

    # 数值水波示意图
    def wave_fig(self):
        a = 0.069
        h = 0.228
        x = np.linspace(-20,20,100)
        temp = np.sqrt((3*a)/(4*h**3))
        ita = a * 1 / np.cosh(temp*x) ** 2

        plt.figure()
        plt.plot(x, ita)
        plt.show()