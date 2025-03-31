import numpy as np
from abc import ABC, abstractmethod

class SearchPattern(ABC):
    """
    Abstract base class for defining search patterns.
    """

    def __init__(self, params, n=25, vel_norm=1.0, dt=0.01):
        assert params.shape[1] == 3, "Parameters need to be three dimensional!"
        # Save the parameters
        self.init_params = params
        self.params = params
        # Create a discrete set of sampling points along tau
        self.tau_dis = np.linspace(0, 1, n)
        # Find the corresponding trajectory points
        self.traj_dis = np.array([self.f(tau) for tau in self.tau_dis])
        # Save the velocity norm and dt
        self.vel_norm = vel_norm
        self.dt=dt

    @abstractmethod
    def f(self, tau):
        """
        Compute the position at point tau along the search pattern.
        """
        pass

    @abstractmethod
    def df(self, tau):
        """
        Compute the velocity at point tau along the search pattern.
        """
        pass

    @abstractmethod
    def ddf(self, tau):
        """
        Compute the velocity at point tau along the search pattern.
        """
        pass

    def reset(self):
        """
        Reset the search pattern to its initial state.
        """
        self.params = self.init_params

        # Find the corresponding trajectory points
        self.traj_dis = np.array([self.f(tau) for tau in self.tau_dis])

    def step_height(self, dh):
        """
        Add a step height to the search pattern.
        """
        # Update height of search pattern
        self.params[-1, 2] += dh

        # Find the corresponding trajectory points
        self.traj_dis = np.array([self.f(tau) for tau in self.tau_dis])

    def find_nearest_tau(self, x, last_tau=-1, iter=2):
        """
        Find the tau \\in [0,1] that correspondes to the point p with the 
        minimal distance to the queried point x
        """
        # If a last tau is given we start the newton's method refinement
        # from that last tau
        if last_tau >= 0:
            tau = last_tau
        # Otherwise we do a brute search over the discrete traj points
        else:
            # Find the vector of squared distances from the current x to the 
            # discrete samples of the trajectory
            distances2 = ((self.traj_dis[0, :] - x[0])**2
                        + (self.traj_dis[1, :] - x[1])**2
                        + (self.traj_dis[2, :] - x[2])**2)
            # Find the best tau
            tau = self.tau_dis[np.argmin(distances2)]
        
        # Define the functions for the newton optimization step
        dg = lambda t: 2*(np.sum((self.f(t) - x) * self.df(t)))
        ddg = lambda t: 2*(np.sum(self.df(t)**2 + (self.f(t) - x) * self.ddf(t)))

        # Run the newton optimization
        for _ in range(iter):
           tau = tau - dg(tau) / ddg(tau)

        # Cap the return value between 0 and 1 and return
        return np.clip(tau, a_min=0, a_max=1)

    def field(self, x, last_tau=-1, kappa=50.0):

        # Find minimum distance trajectory point
        t_min = self.find_nearest_tau(x, last_tau=last_tau)

        # Now compute the normalized distance vector
        distance_vector = (self.f(t_min).flatten() - x)
        distance_norm = np.linalg.norm(distance_vector)

        # Find traj derivative at that index
        derivative = self.df(t_min).flatten()

        # Compute the gradient
        gradient = (np.exp(-distance_norm) * derivative
                    + kappa * distance_vector)

        # Normalize and stretch according to velocity
        return gradient / np.linalg.norm(gradient) * self.vel_norm, t_min

    def get_ref_pos_vel(self, x, last_tau=-1):
        """
        Get the reference position and velocity for point x in space.
        """

        # Find the vector field value at the current position
        gradient, tau_min = self.field(x, last_tau=last_tau)

        # Find the corresponding position and velocity
        p_des = x + self.dt * gradient
        v_des = gradient

        return p_des, v_des, tau_min

class LinearSearchPattern(SearchPattern):
    def __init__(self, params):
        super().__init__(params)
        # params[0, :] Slope; params[1, :] Offset

    def f(self, tau):
        return self.params[1, :] + self.params[0, :] * tau

    def df(self, tau):
        return self.params[0, :]
    
    def ddf(self, tau):
        return 0

class SinusoidalSearchPattern(SearchPattern):
    def __init__(self, params):
        super().__init__(params)

    def f(self, tau):
        return (self.params[3, :]
                + self.params[0, :] * np.sin(2 * np.pi * self.params[1, :] * tau
                                          + self.params[2, :]))

    def df(self, tau):
        return (self.params[0, :] * self.params[1, :] * 2 * np.pi
                * np.cos(2 * np.pi * self.params[1, :] * tau
                                          + self.params[2, :]))
    def ddf(self, tau):
        return (-self.params[0, :] * (self.params[1, :] * 2 * np.pi)**2
                * np.sin(2 * np.pi * self.params[1, :] * tau
                                          + self.params[2, :]))

class CompositeSearchPattern(SearchPattern):
    def __init__(self, patterns):
        self.patterns = patterns
        super().__init__(np.zeros([1, 3]))

    def f(self, tau):
        pos = np.zeros(3)
        for pattern in self.patterns:
            pos += pattern.f(tau)
        return pos

    def df(self, tau):
        vel = np.zeros(3)
        for pattern in self.patterns:
            vel += pattern.df(tau)
        return vel
    
    def ddf(self, tau):
        acc = np.zeros(3)
        for pattern in self.patterns:
            acc += pattern.ddf(tau)
        return acc
    
    def step_height(self, dh):
        for pattern in self.patterns:
            pattern.step_height(dh / len(self.patterns))

        # Find the corresponding trajectory points
        self.traj_dis = np.array([self.f(tau) for tau in self.tau_dis])

    def reset(self):
        """
        Reset the search pattern to its initial state.
        """
        for pattern in self.patterns:
            pattern.params = pattern.init_params

        # Find the corresponding trajectory points
        self.traj_dis = np.array([self.f(tau) for tau in self.tau_dis])