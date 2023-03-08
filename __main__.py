# -*- coding: utf-8 -*-

'''
Author: Lorces
Main program of CFD-HFST
'''

from mesh import *
from ocerm import *
from paperfig import *


# 1 - Mesh file generation
# 2 - Pre-process for include files
# 3 - Analysis of numerical results
# 4 - Visualization for IBM module

tool_mode = 4


if tool_mode == 1:
    bd1 = BdCreate()
    boundary = bd1.add_arc([0,0], 0.075, [0, 360], 500)

    rect1 = MeshFile('CYT1')
    # rect1.rect_mesh(-23, 17, 801, -0.05, 0.05, 3)

    unst1 = MeshFile('CYIN')
    # unst1.unst_mesh(boundary)

    custom1 = MeshCustom('USOU','USIN')
    # custom1.refine_mesh()
    # custom1.custom_mesh()

    ib1 = IbFile()
    # ib1.ib_2D(boundary)
    # ib1.ib_3D(3)

elif tool_mode == 2:
    u = 0.0 * 0.04
    h = 0.16
    kbh = 161

    kb_tool_mode = 0
    kb_down = 0.001
    kb_mult = 1.2

    inc_create = IncludeCreate()
    kb,per = inc_create.set_kb(kbh,h,kb_down,kb_mult,kb_tool_mode)
    qIn = inc_create.create_qbc(u, kb, h)

    gauge_num = 101
    gauge_x, gauge_y = -0.9, 0
    gauge_dx = 0.04
    inc_create.create_gauge(gauge_num,gauge_dx,gauge_x)

    iteration = 0
    meshtype = 1

    ocerm1 = IncludeModify()

    if iteration == 0:
        ocerm1.modify_grd(kb,h,per)
        ocerm1.modify_cuv()
        ocerm1.modify_inf(kb,gauge_num)

        if meshtype == 1:
            ocerm1.modify_qbc(qIn,kb)
            ocerm1.modify_ebc('0.000000')
        elif meshtype == 2:
            in_start, in_end = 1,20
            out_start, out_end = 1181,1200
            ocerm1.modify_qbc_unst(in_start,in_end,qIn,kb)
            ocerm1.modify_ebc_unst(out_start,out_end,'0.000000')

elif tool_mode == 3:
    u_inf = 0.1
    d = 0.039
    gauge_num = 150

    case1 = VisResult()

    # case1.u_Cp()
    # case1.u_mid(u_inf,d)
    # case1.u_boundary()
    # case1.wave_fig()
    # case1.u_fft(gauge_num)

elif tool_mode == 4:
    circle = [0, 0, 0.02]  # xr,yr,r,r2
    wmr = 0.003

    ibm_case1 = VisIBM()

    # ibm_case1.ibm_gc_3D()
    # ibm_case1.ibm_image_2D()
    # ibm_case1.ibm_wm_2D(wmr, circle)
    # ibm_case1.ibm_df_2D(circle)
    # ibm_case1.ibm_ani_2D()
    # ibm_case1.ibm_mesh(4)
    # ibm_case1.ibm_vec()




