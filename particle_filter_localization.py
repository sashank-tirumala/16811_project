import numpy as np 
import pandas as pd 
from scipy import stats
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

def initialize_particles(num_particles,init, xn, yn, thetan):
    x_ini = init[0,1]
    y_ini = init[0,2]
    theta_ini = init[0,3]
    particles = np.zeros((num_particles, 3))
    particles[:,0] = np.random.normal(x_ini, xn, num_particles)
    particles[:,1] = np.random.normal(y_ini, yn, num_particles)
    particles[:,2] = np.random.normal(theta_ini, thetan, num_particles)
    return particles

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

def prediction_update(time, particles, odom_data, v_noise, w_noise):
    """
    Implements prediction update equations of a Kalman Filter
    """
    delta_t = odom_data["Time"] - time
    lin_vel = odom_data["c1"]
    ang_vel = odom_data["c2"]
    for particle in particles:
            lin_vel_p = np.random.normal(lin_vel, v_noise, 1)
            ang_vel_p = np.random.normal(ang_vel, w_noise, 1)
            particle[0] += lin_vel_p * np.cos(particle[2]) * delta_t
            particle[1] += lin_vel_p * np.sin(particle[2]) * delta_t
            particle[2] += ang_vel_p * delta_t

            if (particle[2] > np.pi):
                particle[2] -= 2 * np.pi
            elif (particle[2] < -np.pi):
                particle[2] += 2 * np.pi

    return particles, odom_data["Time"]

def measurement_update(time, particles, measurement_data, local_map, range_noise, bearing_noise):
    delta_t = measurement_data["Time"] - time
    measure_range = measurement_data['c1']
    measure_bearing = measurement_data['c2']
    df = local_map.loc[local_map['subject'] == measurement_data['subject']]
    weights = np.zeros(len(particles))
    if(df.empty):
        return particles, measurement_data["Time"]
    else:
        pass
    x_l = local_map.loc[local_map['subject'] == measurement_data['subject']].iloc[0,1]
    y_l = local_map[local_map['subject'] == measurement_data['subject']].iloc[0,2]
    for i in range(len(particles)):
        x_curr = particles[i][0]
        y_curr = particles[i][1]
        theta_curr = particles[i][2]
        q = (x_l - x_curr) * (x_l - x_curr) + (y_l - y_curr) * (y_l - y_curr)
        range_expected = np.sqrt(q)
        bearing_expected = np.arctan2(y_l - y_curr, x_l - x_curr) - theta_curr
        range_error = measure_range - range_expected
        bearing_error = measure_bearing - bearing_expected
        prob_range = stats.norm(0, range_noise).pdf(range_error)
        prob_bearing = stats.norm(0, bearing_noise).pdf(bearing_error)
        weights[i] = prob_range * prob_bearing
    
    if(np.sum(weights) == 0):
        weights = np.ones_like(weights)
    weights /= np.sum(weights)
    new_idexes = np.random.choice(len(particles), len(particles), replace = True, p = weights)
    particles = particles[new_idexes]
    return particles, measurement_data["Time"]

def calc_state( particles, time):
    state = np.mean(particles, axis = 0)
    return np.array([[time, state[0], state[1], state[2]]])

def particle_filter_update(states, data, df_map):
    x_noise, y_noise, theta_noise, v_noise, w_noise = [0.1, 0.1, 0.1, 0.2, 0.2]
    range_noise, bearing_noise = [0.1, 0.1]
    particles = initialize_particles(10,states, x_noise, y_noise, theta_noise)
    time = states[0,0]
    for index, row in data.iterrows():
        if(row['subject'] == -1):
            particles, time = prediction_update(time, particles, row, v_noise, w_noise)
        else:
            particles, time = measurement_update(time, particles, row, df_map, range_noise, bearing_noise)
        new_val = calc_state(particles, time)
        states = np.append(states, new_val, axis=0)
    return states

if(__name__ == "__main__"):
    num_pts = 20000 #Considering first 20000 points for measurement
    df = pd.read_csv("0.Dataset1/Groundtruth.dat", delim_whitespace=True)
    df_odom = pd.read_csv("0.Dataset1/Odometry.dat", delim_whitespace=True)
    df_map = pd.read_csv("0.Dataset1/Landmark_Groundtruth.dat", delim_whitespace=True)
    df_meas = pd.read_csv("0.Dataset1/Measurement.dat", delim_whitespace=True)
    time_end= df.iloc[num_pts]['Time']
    data = load_data(df_odom, df_meas, time_end)
    states = np.array([[df.iloc[0,0], df.iloc[0,1], df.iloc[0,2], df.iloc[0,3]]])
    states = particle_filter_update(states, data, df_map)
    x_gt = df['x']
    y_gt = df['y']
    ori_gt = df['orientation']
    x_est, y_est, ori_est = odom_est([x_gt[0], y_gt[0], ori_gt[0]], df_odom.loc[df_odom['Time']<time_end], 0.120)
    fig = plt.figure()
    plt.plot(x_gt[:num_pts], y_gt[:num_pts], label = "Ground Truth")
    plt.plot(states[:,1], states[:,2], 'r', label ="Particle Filter Estimate")
    plt.plot(x_est, y_est, 'g', label = "Odometry Estimate")
    plt.legend()
    plt.show()
