import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParticleFilter:
    def __init__(self, num_particles=1000, pos_std=20,
                 process_std_x=50, process_std_y=20,
                 obs_std=50):
        self.num_particles = num_particles

        # Initialize particle positions with Gaussian noise around initial position
        self.particles = torch.randn(num_particles, 2, device=device) * pos_std

        # Initialize uniform weights
        self.weights = torch.ones(num_particles, device=device) / num_particles

        self.process_std_x = process_std_x  # Standard deviation of process noise
        self.process_std_y = process_std_y
        self.obs_std = obs_std          # Standard deviation of observation model

    # The following function is created only to keep some consistency in
    # class methods across different filters. Could have done the following
    # within __init__.
    def initialize(self, position):
        self.particles += torch.tensor(position, device=device)

    def predict(self):
        # Apply random walk (can be replaced with a motion model)
        noise_x = torch.randn(self.num_particles, device=device) * self.process_std_x
        noise_y = torch.randn(self.num_particles, device=device) * self.process_std_y
        noise = torch.concat((noise_x.unsqueeze(1),
                              noise_y.unsqueeze(1)), dim=1)
        self.particles += noise

    def update(self, observation):
        # Compute distance to observation
        dists = torch.norm(self.particles - observation, dim=1)

        # Compute Gaussian likelihood (unnormalized)
        likelihoods = torch.exp(-0.5 * (dists / self.obs_std) ** 2)

        # Update weights and normalize
        self.weights = likelihoods
        self.weights /= self.weights.sum() + 1e-12

    def resample(self):
        # Systematic resampling
        cdf = torch.cumsum(self.weights, dim=0)
        step = 1.0 / self.num_particles
        start = torch.rand(1).item() * step
        points = torch.arange(start, 1.0, step, device=device)
        indices = torch.searchsorted(cdf, points)
        indices[indices == self.num_particles] -= 1 # To avoid indexing

        self.particles = self.particles[indices]
        self.weights.fill_(1.0 / self.num_particles)

    def estimate(self):
        # Return weighted mean estimate. torch.sum instead of torch.mean
        # because already normalized.

        return (self.particles * self.weights.unsqueeze(1)).sum(0)