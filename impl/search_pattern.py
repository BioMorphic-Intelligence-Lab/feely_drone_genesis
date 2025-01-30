import numpy as np
from abc import ABC, abstractmethod

class SearchPattern(ABC):
    """
    Abstract base class for defining search patterns.
    """

    @abstractmethod
    def f(self, t):
        """
        Compute the position at time t.
        """
        pass

    @abstractmethod
    def df(self, t):
        """
        Compute the velocity at time t.
        """
        pass

    def get_ref_pos_vel(self, t):
        """
        Get the reference position and velocity.
        """
        return self.f(t), self.df(t)

class LinearSearchPattern(SearchPattern):
    def __init__(self, slope, offset):
        super().__init__()
        assert len(slope) == 3, "Slope must be 3D!"
        assert len(offset) == 3, "Offset must be 3D!"
        self.slope = np.array(slope)
        self.offset = np.array(offset)

    def f(self, t):
        return self.offset + self.slope * t

    def df(self, t):
        return self.slope

class SinusoidalSearchPattern(SearchPattern):
    def __init__(self, amplitude, frequency, phaseshift, offset):
        super().__init__()
        assert len(amplitude) == 3, "Amplitude must be 3D!"
        assert len(frequency) == 3, "Frequency must be 3D!"
        assert len(phaseshift) == 3, "Phaseshift must be 3D!"
        assert len(offset) == 3, "Offset must be 3D!"
        self.amplitude = np.array(amplitude)
        self.frequency = np.array(frequency)
        self.phaseshift = np.array(phaseshift)
        self.offset = np.array(offset)

    def f(self, t):
        return (self.offset
                + self.amplitude * np.sin(2 * np.pi * self.frequency * t
                                          + self.phaseshift))

    def df(self, t):
        return (self.amplitude * self.frequency * 2 * np.pi
                * np.cos(2 * np.pi * self.frequency * t
                                          + self.phaseshift))

class CompositeSearchPattern(SearchPattern):
    def __init__(self, patterns):
        self.patterns = patterns

    def f(self, t):
        pos = np.zeros(3)
        for pattern in self.patterns:
            pos += pattern.f(t)
        return pos

    def df(self, t):
        vel = np.zeros(3)
        for pattern in self.patterns:
            vel += pattern.df(t)
        return vel