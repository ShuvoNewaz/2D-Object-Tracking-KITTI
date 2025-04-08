import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KalmanFilter:
    def __init__(self, dt=0.1, sigma_ax=20, sigma_ay=10):
        self.dt = dt

        # State transition matrix (4x4)
        self.A = torch.tensor([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0 ],
                               [0, 0, 0, 1 ]],
                               dtype=torch.float32, device=device)

        # Observation matrix (2D position only)
        self.H = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0]
                               ],
                               dtype=torch.float32, device=device)

        # Process noise covariance Q (4x4)
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        self.Q = torch.tensor([[0.25 * dt4 * sigma_ax ** 2, 0, 0.5 * dt3 * sigma_ax ** 2, 0],
                               [0, 0.25 * dt4 * sigma_ay ** 2, 0, 0.5 * dt3 * sigma_ay ** 2],
                               [0.5 * dt3 * sigma_ax ** 2, 0, dt2 * sigma_ax ** 2, 0],
                               [0, 0.5 * dt3 * sigma_ay ** 2, 0, dt2 * sigma_ay ** 2]],
                               dtype=torch.float32, device=device)

        # Observation noise covariance R
        self.R = torch.eye(2, device=device) * 1.0  # adjustable

        # State estimate (4D) and covariance (4x4)
        self.x = torch.zeros(4, 1, device=device)
        self.P = torch.eye(4, device=device)

    def initialize(self, position, velocity=None):
        """Initialize filter with position [x, y] and optional velocity [vx, vy]."""
        self.x[0, 0] = position[0]
        self.x[1, 0] = position[1]
        if velocity is not None:
            self.x[2, 0] = velocity[0]
            self.x[3, 0] = velocity[1]

    def predict(self):
        """Prediction step."""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        """Update step with observation z (2x1 torch tensor)."""
        y = z.view(2, 1) - (self.H @ self.x)                  # Innovation
        S = self.H @ self.P @ self.H.T + self.R               # Innovation covariance
        K = self.P @ self.H.T @ torch.linalg.inv(S)           # Kalman gain
        self.x = self.x + K @ y                               # Updated state
        I = torch.eye(self.P.shape[0], device=device)
        self.P = (I - K @ self.H) @ self.P                    # Updated covariance

    def estimate(self):
        """Returns current state estimate as (x, y, vx, vy)."""
        
        return self.x.flatten()
