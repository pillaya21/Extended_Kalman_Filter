import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init = data['x_init']  # initial x position [m]
y_init = data['y_init']  # initial y position [m]
th_init = data['th_init']  # initial theta position [rad]

# input signal
v = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.008  # translation velocity variance
om_var = 6 # rotational velocity variance
r_var = 0.00001  # range measurements variance
b_var = 0.001  # bearing measurement variance

Q_km = np.diag([v_var, om_var])  # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init])  # initial state
P_est[0] = np.diag([1, 1, 0.1])  # initial state covariance


# Wraps angle to (-pi,pi] range
def wraptopi(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x = x - 2 * np.pi
    elif x < -np.pi:
        x += 2 * np.pi
    return x


def measurement_model(x_check, lk):
    A = lk[0] - x_check[0] - d[0] * np.cos(x_check[2])
    B = lk[1] - x_check[1] - d[0] * np.sin(x_check[2])
    y_check = np.array([[np.sqrt(A ** 2 + B ** 2)], [np.arctan2(B, A) - x_check[2]]])
    return y_check


def measurement_update(lk, rk, bk, P_check, x_check):
    # 1. Compute measurement Jacobian
    x_check[2] = wraptopi(x_check[2])
    A = lk[0] - x_check[0] - d[0] * np.cos(x_check[2])
    B = lk[1] - x_check[1] - d[0] * np.sin(x_check[2])
    C = (lk[0] - x_check[0])
    D = lk[1] - x_check[1]
    dist = np.sqrt(A ** 2 + B ** 2)
    H = np.array([[-A / dist, -B / dist, (d[0] * (C * np.sin(x_check[2]) + D * np.cos(x_check[2]))) / dist],
                  [B / (dist ** 2), -A / (dist ** 2),
                   (D * (d[0] * np.sin(x_check[2]) + -D) + C * (d[0] * np.cos(x_check[2]) - C)) / (dist ** 2)]])
    M = np.eye(2)
    R = cov_y
    MRM = np.linalg.multi_dot([M, R, M.T])
    HPH = np.linalg.multi_dot([H, P_check, H.T])
    total = MRM + HPH
    totalinv = np.linalg.inv(total)

    # 2. Compute Kalman Gain
    K = np.linalg.multi_dot([P_check, H.T, totalinv])

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    y_check = measurement_model(x_check, lk)
    y_check[1, 0] = wraptopi(y_check[1, 0])
    bk = wraptopi(bk)
    y = np.array([[rk], [bk]])
    inno = y - y_check
    add = np.dot(K, inno).reshape((3,))
    x_check = x_check + add
    x_check[2] = wraptopi(x_check[2]);

    # 4. Correct covariance
    P_check = np.dot((np.eye(3) - np.dot(K, H)), P_check)

    return x_check, P_check


def motion_model(x_check, k, delta_t):
    A = np.array([[(np.cos(x_check[2])), 0], [(np.cos(x_check[2])), 0], [0, 1]])
    B = np.array([[v[k]], [om[k]]])
    add = delta_t * np.dot(A, B).reshape((3,))
    x_check = x_check + add
    x_check[2] = wraptopi(x_check[2])
    return x_check


for k in range(1, len(t)):  # start at 1 because we've set the initial prediction

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    x_check = np.zeros((3,))
    x_check[0] = x_est[k - 1, 0]
    x_check[1] = x_est[k - 1, 1]
    x_check[2] = x_est[k - 1, 2]
    P_check = P_est[k - 1, :, :]
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    x_check = motion_model(x_check, k, delta_t)

    # 2. Motion model jacobian with respect to last state
    F_km = np.array(
        [[1, 0, -delta_t * v[k - 1] * np.sin(x_check[2])], [0, 1, delta_t * v[k - 1] * np.cos(x_check[2])], [0, 0, 1]])

    # 3. Motion model jacobian with respect to noise
    L_km = delta_t * np.array([[np.cos(x_check[2]), 0], [np.sin(x_check[2]), 0], [0, 1]])
    noise = np.linalg.multi_dot([L_km, Q_km, L_km.T])
    motion_c = np.linalg.multi_dot([F_km, P_check, F_km.T])
    P_check = noise + motion_c
    # 4. Propagate uncertainty

    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()

with open('submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)
