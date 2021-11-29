import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def load_data(odom_data, measurement_data, time_end):
    """
    Create a single dataframe that consists of odometry and measurement, which is also ordered based on timestamp.
    """
    odom_data = odom_data.loc[df_odom['Time']<time_end]
    odom_data = odom_data.rename(columns={"forward_velocity":"c1", "angular_velocity":"c2"})
    odom_data["subject"] = -1
    measurement_data = measurement_data.loc[measurement_data['Time']<time_end]
    measurement_data = measurement_data.rename(columns={"range":"c1", "bearing":"c2"})
    dat = pd.concat([odom_data, measurement_data])
    dat = dat.sort_values(by=['Time'])
    print(dat.columns)
    return dat

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

def prediction_update(states, odom_data, sigma, R):
    """
    Implements prediction update equations of a Kalman Filter
    """
    
    time, x_prev, y_prev, ori_prev = states[-1, :]
    delta_t = odom_data["Time"] - time
    lin_vel = odom_data["c1"]
    ang_vel = odom_data["c2"]
    if(delta_t < 0.001):
        return None, sigma
    #process update
    x_cur = x_prev + lin_vel*np.cos(ori_prev)*delta_t
    y_cur = y_prev + lin_vel*np.sin(ori_prev)*delta_t
    ori_cur = ori_prev + ang_vel*delta_t

    #Covariance update
    G_1 = np.array([1, 0, - lin_vel * delta_t * np.sin(ori_prev)])
    G_2 = np.array([0, 1, lin_vel * delta_t * np.cos(ori_prev)])
    G_3 = np.array([0, 0, 1])
    G = np.array([G_1, G_2, G_3])
    sigma = G.dot(sigma).dot(G.T) + R
    return np.array([[odom_data["Time"], x_cur, y_cur, ori_cur]]), sigma

def measurement_update(states, measurement_data, local_map, sigma, Q):
    """
    Implements measurement update of a Kalman Filter
    """
    time, x_prev, y_prev, ori_prev = states[-1, :]
    delta_t = measurement_data["Time"] - time
    measure_range = measurement_data['c1']
    measure_bearing = measurement_data['c2']
    #Calculate expected measurement
    df = local_map.loc[local_map['subject'] == measurement_data['subject']]
    if(df.empty):
        return None, sigma
    else:
        pass
    x_l = local_map.loc[local_map['subject'] == measurement_data['subject']].iloc[0,1]
    y_l = local_map[local_map['subject'] == measurement_data['subject']].iloc[0,2]
    x_t = states[-1][1]
    y_t = states[-1][2]
    theta_t = states[-1][3]
    q = (x_l - x_t) * (x_l - x_t) + (y_l - y_t) * (y_l - y_t)
    range_expected = np.sqrt(q)
    bearing_expected = np.arctan2(y_l - y_t, x_l - x_t) - theta_t

    #Calculate Linearized Jacobian for measurement
    H_1 = np.array([-(x_l - x_t) / np.sqrt(q), -(y_l - y_t) / np. sqrt(q), 0])
    H_2 = np.array([(y_l - y_t) / q, -(x_l - x_t) / q, -1])
    H_3 = np.array([0, 0, 0])
    H = np.array([H_1, H_2, H_3])

    #Calculate the Kalman Gain
    S_t = H.dot(sigma).dot(H.T) + Q
    K = sigma.dot(H.T).dot(np.linalg.inv(S_t))

    #Calculate new state data
    difference = np.array([measure_range - range_expected, measure_bearing - bearing_expected, 0])
    innovation = K.dot(difference)
    x_cur = x_t + innovation[0]
    y_cur = y_t + innovation[1]
    theta_cur = theta_t + innovation[2]

    #update covariance
    sigma = (np.identity(3) - K.dot(H)).dot(sigma)
    return np.array([[measurement_data["Time"], x_cur, y_cur, theta_cur]]), sigma

def kalman_filter_update(states, data, local_map):
    #Initialize Constants
    sigma = np.diagflat([1e-10, 1e-10, 1e-10]) 
    R = np.diagflat(np.array([1.0, 1.0, 10.0])) ** 2
    Q = np.diagflat(np.array([30, 30, 30])) ** 2

    for index, row in data.iterrows():
        if(row['subject'] == -1):
            new_val, sigma = prediction_update(states, row, sigma, R)
        else:
            new_val, sigma = measurement_update(states, row, local_map, sigma, Q)
        if(new_val is not None):
            states = np.append(states, new_val, axis=0)
    return states

def odom_est(init_pts, odom_data, delta_t):
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
    df_map = pd.read_csv("0.Dataset1/Landmark_Groundtruth.dat", delim_whitespace=True)
    df_meas = pd.read_csv("0.Dataset1/Measurement.dat", delim_whitespace=True)
    time_end= df.iloc[num_pts]['Time']
    data = load_data(df_odom, df_meas, time_end)
    states = np.array([[df.iloc[0,0], df.iloc[0,1], df.iloc[0,2], df.iloc[0,3]]])
    states = kalman_filter_update(states, data, df_map)
    x_gt = df['x']
    y_gt = df['y']
    ori_gt = df['orientation']
    x_est, y_est, ori_est = odom_est([x_gt[0], y_gt[0], ori_gt[0]], df_odom.loc[df_odom['Time']<time_end], 0.120)
    fig = plt.figure()
    plt.plot(x_gt[:num_pts], y_gt[:num_pts], label = "Ground Truth")
    plt.plot(states[:,1], states[:,2], 'r', label ="Kalman Estimate")
    plt.plot(x_est, y_est, 'g', label = "Odometry Estimate")
    plt.legend()
    plt.show()




    

