
# coding: utf-8

import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

fs = 200   # 输入频率（回传速率）

df = pd.read_excel(r"你的文件路径",sheet_name="你的表名")

# 用你表中的列名
ax_g = df['AX'].to_numpy()
ay_g = df['AY'].to_numpy()
az_g = df['AZ'].to_numpy()


# 加权滤波器

def Hh(f1):
    w1 = 2 * np.pi * f1
    num = [1, 0, 0]
    den = [1, np.sqrt(2)*w1, w1**2]
    return num, den

def Hl(f2):
    w2 = 2 * np.pi * f2
    num = [w2**2]
    den = [1, np.sqrt(2)*w2, w2**2]
    return num, den

def Ht(f3, f4, Q4):
    w4 = 2 * np.pi * f4

    # f3 = ∞ 
    if np.isinf(f3):
        num = [1]
    else:
        w3 = 2 * np.pi * f3
        num = [1/w3, 1]   

    den = [1/(w4**2), 1/(Q4*w4), 1]
    return num, den

def Hs(f5, f6, Q5, Q6):
    w5 = 2 * np.pi * f5
    w6 = 2 * np.pi * f6
    num = [1/(w5**2), 1/(Q5*w6), 1]
    den = [1/(w6**2), 1/(Q6*w6), 1]
    gain = (w5 / w6)**2
    num = [gain * n for n in num]
    return num, den

def cascade(filters):
    num, den = [1], [1]
    for n, d in filters:
        num = np.convolve(num, n)
        den = np.convolve(den, d)
    return num, den




def iso2631_weighting_filter(weighting, fs):

    if weighting == 'Wk':
        filters = [
            Hh(0.4),
            Hl(100),
            Ht(12.5, 12.5, 0.63),
            Hs(2.37, 3.35, 0.91, 0.91)
        ]

    elif weighting == 'Wd':
        filters = [
            Hh(0.4),
            Hl(100),
            Ht(2.0, 2.0, 0.63)
        ]

    elif weighting == 'Wf':
        filters = [
            Hh(0.08),
            Hl(0.63),
            Ht(np.inf, 0.25, 0.86),   
            Hs(0.0625, 0.1, 0.80, 0.80)
        ]

    else:
        raise ValueError("weighting must be 'Wk', 'Wd' or 'Wf'")

    num_a, den_a = cascade(filters)
    b, a = signal.bilinear(num_a, den_a, fs)
    return b, a



def rms(x):
    return np.sqrt(np.mean(x**2))

def msdv(x, fs):
    dt = 1 / fs
    return (np.sum(x**2) * dt) ** 0.5



def iso2631_axis(acc_g, fs, weighting):
    g0 = 9.80665
    acc = acc_g * g0  # 将单位g转换为m/s^2

    b, a = iso2631_weighting_filter(weighting, fs)
    aw = signal.filtfilt(b, a, acc)

    return aw, rms(aw), msdv(aw, fs)




def decimate_to_20hz(x, fs):
    if fs == 20:
        return x, fs

    factor = int(fs / 20)
    x_ds = signal.decimate(x, factor, ftype='iir', zero_phase=True)
    return x_ds, 20





_, rms_x, msdv_x = iso2631_axis(ax_g, fs, 'Wd')  # X轴 rms 用 Wd 加权
_, rms_y, msdv_y = iso2631_axis(ay_g, fs, 'Wd')  # Y轴 rms 用 Wd 加权
_, rms_zk, msdv_z_wk = iso2631_axis(az_g, fs, 'Wk')# Z轴rms 用 Wk 加权


az_ds, fs_wf = decimate_to_20hz(az_g, fs)#若200hz则降频到20hz再计算，用200hz计算会出现NaN错误
_, rms_zf, msdv_z_wf = iso2631_axis(az_ds, fs_wf, 'Wf')# Z轴 MSDV 用 Wf 加权
    
    
    
# 输出z轴Wf加权的MSDV
print(f"Z 轴 MSDV (Wf 加权) = {msdv_z_wf:.4f} m/s^(1.5)")

# 计算总合成 RMS
rms_total1 = np.sqrt(rms_x**2 + rms_y**2 + rms_zk**2)
print(f"Total RMS = {rms_total1:.4f} m/s²")


