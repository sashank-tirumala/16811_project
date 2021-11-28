import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
def odom_est(init_pts, odom_data, delta_t):
    print(init_pts)
    x_est = []
    y_est = []
    ori_est = []
    x_cur = init_pts[0]
    y_cur = init_pts[1]
    ori_cur = init_pts[2]
    for index, data in odom_data.iterrows():
        lin_vel = data["forward_velocity"]
        ang_vel = data["angular_velocity"]
        x_est.append(x_cur)
        y_est.append(y_cur)
        ori_est.append(ori_cur)
        x_cur = x_cur + lin_vel*np.cos(ori_cur)*delta_t
        y_cur = y_cur + lin_vel*np.sin(ori_cur)*delta_t
        ori_cur = ori_cur + ang_vel*delta_t
    return np.array(x_est), np.array(y_est), np.array(ori_est)
if(__name__ == "__main__"):
    num_pts = 20000 #Considering first 20000 points for measurement
    df = pd.read_csv("0.Dataset1/Groundtruth.dat", delim_whitespace=True)
    df_odom = pd.read_csv("0.Dataset1/Odometry.dat", delim_whitespace=True)
    print(df_odom.head())
    x_gt = df['x']
    y_gt = df['y']
    ori_gt = df['orientation']
    time_end= df.iloc[num_pts]['Time']
    x_est, y_est, ori_est = odom_est([x_gt[0], y_gt[0], ori_gt[0]], df_odom.loc[df_odom['Time']<time_end], 0.120)
    print(x_est.shape, x_gt.shape)
    fig = plt.figure()
    plt.plot(x_gt[:num_pts], y_gt[:num_pts])
    plt.plot(x_est[:num_pts], y_est[:num_pts], 'r')
    plt.show()

