import numpy as np

class PoseCtrl:

    def __init__(self, m_total, dt,
                    g=np.array([0, 0, -9.81]),
                    kp=500, ki=25,
                    kd=250, ky=200, komega=200):
        
        # Remember time constant
        self.dt = dt
        # Remember gravity
        self.g = g
        # Remember total system mass for feedforward
        self.m = m_total
        # Remember gains for Positional PID controller
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Remember gains for 
        self.ky = ky
        self.komega = komega

        # Init integral errors
        self.p_err_I = np.zeros(3)
    
    def reset(self):
        self.p_err_I = np.zeros(3)

    def pos_ctrl(self, p_des, p, v_des, v):
        p_err = p_des - p
        v_err = v_des - v
        
        self.p_err_I += self.dt * p_err

        return -self.g * self.m + self.kp * p_err + self.ki * self.p_err_I + self.kd * v_err
    
    def yaw_ctrl(self, yaw_des, yaw, omega_des, omega):
        yaw_err = yaw_des - yaw
        omega_err = omega_des - omega

        return self.ky * yaw_err + self.komega * omega_err
