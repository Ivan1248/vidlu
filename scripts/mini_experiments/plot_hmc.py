# The code is based on code from
# https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch
# and https://github.com/ColCarroll/minimc

from autograd import grad
import autograd.numpy as np
import autograd.scipy.stats as st
import scipy.stats


def leapfrog(f, x, p, step_count, step_size):
    x, p = np.copy(x), np.copy(p)
    p -= step_size * f(x) / 2  # half step
    for _ in range(step_count):
        x += step_size * p  # whole step
        p -= step_size * f(x)  # whole step
    x += step_size * p  # whole step
    p -= step_size * f(x) / 2  # half step
    return x, p


def hmc(U, T, x_start, p_dist, L=100, epsilon=0.1):
    K = lambda p: -p_dist.logpdf(p)

    for _ in range(T):
        p_start = p_dist.sample()
        # Integrate over our path to get a new position and momentum
        x_prop, p_prop = leapfrog(grad(U), x_start, p_start, L, epsilon)
        # Negate momentum to make the proposal symmetric
        p_prop = -p_prop
        # Metropolis acceptance criterion
        H_curr = U(x_start) + K(p_start)
        H_prop = U(x_prop) + K(p_prop)
        if np.random.rand() < np.exp(H_curr - H_prop):
            yield x_prop
        else:
            yield x_start


class Normal2:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def logpdf(self, x):
        return np.sum(st.norm.logpdf(x, self.mean, self.std))

    def sample(self):
        return scipy.stats.norm(self.mean, self.std).rvs(2)


U = lambda x: -np.sum(st.norm.logpdf(x, 1, 0.5))

samples = [x for x in hmc(U, 10, x_start=np.array([0., 0.]), p_dist=Normal2(0, 1))]


def leapfrog2(q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.
    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be
    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)
    positions, momentums = [np.copy(q)], [np.copy(p)]
    stages = [[np.copy(q), np.copy(p)]]

    velocity = dVdq(q)
    for _ in np.arange(np.round(path_len / step_size)):
        p -= step_size * velocity / 2  # half step
        stages.append([np.copy(q), np.copy(p)])
        q += step_size * p  # whole step
        stages.append([np.copy(q), np.copy(p)])
        positions.append(np.copy(q))
        velocity = dVdq(q)
        p -= step_size * velocity / 2  # half step
        stages.append([np.copy(q), np.copy(p)])
        momentums.append(np.copy(p))

    # momentum flip at end
    return q, -p, np.array(positions), np.array(momentums), np.array(stages)


import scipy.stats as st


def hamiltonian_monte_carlo2(
        n_samples,
        potential,
        initial_position,
        path_len=1,
        step_size=0.1,
        integrator=leapfrog2,
        do_reject=True,
):
    """Run Hamiltonian Monte Carlo sampling.
    Parameters
    ----------
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.
    integrator: callable
        Integrator to use, from `integrators_slow.py`
    do_reject: boolean
        Turn off metropolis correction. Not valid MCMC if False!
    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    initial_position = np.array(initial_position)
    # negative_log_prob = lambda q: potential(q)[0]  # NOQA
    # dVdq = lambda q: potential(q)[1]  # NOQA
    dVdq = grad(potential)

    # collect all our samples in a list
    samples = [initial_position]
    sample_positions, sample_momentums = [], []
    accepted = []
    p_accepts = []

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want 100 x 10 momentum draws
    # we can do this in one call to np.random.normal, and iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    for p0 in tqdm(momentum.rvs(size=size)):
        # Integrate over our path to get a new position and momentum
        q_new, p_new, positions, momentums, _ = integrator(
            samples[-1], p0, dVdq, path_len=path_len, step_size=step_size
        )
        sample_positions.append(positions)
        sample_momentums.append(momentums)

        # Check Metropolis acceptance criterion
        start_log_p = potential(samples[-1]) - np.sum(
            momentum.logpdf(p0)
        )
        new_log_p = potential(q_new) - np.sum(momentum.logpdf(p_new))
        energy_change = start_log_p - new_log_p
        p_accept = np.exp(energy_change)

        if np.random.rand() < p_accept:
            samples.append(q_new)
            accepted.append(True)
        else:
            if do_reject:
                samples.append(np.copy(samples[-1]))
            else:
                samples.append(q_new)
            accepted.append(False)
        p_accepts.append(p_accept)

    return (
        np.array(samples[1:]),
        np.array(sample_positions),
        np.array(sample_momentums),
        np.array(accepted),
        np.array(p_accepts),
    )


def neg_log_mvnormal(mu, sigma):
    """Use a Cholesky decomposition for more careful work."""

    def logp(x):
        k = mu.shape[0]
        return (
                       k * np.log(2 * np.pi)
                       + np.log(np.linalg.det(sigma))
                       + np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), x - mu)
               ) * 0.5

    return logp


from autograd.scipy.special import logsumexp


def mixture(neg_log_probs, probs):
    """Log probability of a mixture of probabilities.
    neg_log_probs should be an iterator of negative log probabilities
    probs should be an iterator of floats of the same length that sums to 1-ish
    """
    probs = np.array(probs) / np.sum(probs)
    assert len(neg_log_probs) == probs.shape[0]

    def logp(x):
        return -logsumexp(np.log(probs) - np.array([logp(x) for logp in neg_log_probs]))

    return logp


import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def neg_log_p_to_img(neg_log_p, extent=(-3, 3, -3, 3), num=100):
    X, Y = np.meshgrid(np.linspace(*extent[:2], num), -np.linspace(*extent[2:], num))
    Z = np.array([np.exp(-neg_log_p(j)) for j in np.array((X.ravel(), Y.ravel())).T]).reshape(
        X.shape)
    return Z, extent


###

def plot1():
    np.random.seed(1)

    neg_log_p = neg_log_mvnormal(np.zeros(2), np.eye(2))
    dVdq = grad(neg_log_p)

    positions, momentums = [], []
    for _ in trange(3):
        q, p = np.random.randn(2, 2)
        _, _, q, p, _ = leapfrog2(q, p, dVdq, 2 * np.pi, 0.1)
        positions.append(q)
        momentums.append(p)

    fig, axes = plt.subplots(ncols=len(positions), figsize=(7 * len(positions), 7))

    steps = slice(None, None, 4)

    Z, extent = neg_log_p_to_img(neg_log_p, (-3, 3, -3, 3), num=200)

    for idx, (ax, q, p) in enumerate(zip(axes.ravel(), positions, momentums)):
        ax.imshow(Z, alpha=0.9, extent=extent, cmap='bone_r', origin='upper')
        ax.quiver(q[steps, 0], q[steps, 1], p[steps, 0], p[steps, 1], headwidth=6, scale=60,
                  headlength=7, color='C1')
        ax.plot(q[:, 0], q[:, 1], '-', lw=3, color='C1')
    # plt.savefig('normal_leapfrog.png')
    plt.show()


###

def plot2():
    np.random.seed(7)

    mu1 = np.ones(2)
    cov1 = 0.5 * np.array([[1., 0.7],
                           [0.7, 1.]])
    mu2 = -mu1
    cov2 = 0.2 * np.array([[1., -0.6],
                           [-0.6, 1.]])

    mu3 = np.array([-1., 2.])
    cov3 = 0.3 * np.eye(2)

    neg_log_p = mixture(
        [neg_log_mvnormal(mu1, cov1), neg_log_mvnormal(mu2, cov2), neg_log_mvnormal(mu3, cov3)],
        [0.3, 0.3, 0.4])
    dVdq = grad(neg_log_p)

    positions, momentums = [], []
    for _ in trange(4):
        q, p = np.random.randn(2, 2)
        _, _, q, p, _ = leapfrog2(q, p, dVdq, 4 * np.pi, 0.1)
        positions.append(q)
        momentums.append(p)

    fig, axes = plt.subplots(ncols=len(positions), figsize=(7 * len(positions), 7))

    steps = slice(None, None, 4)

    Z, extent = neg_log_p_to_img(neg_log_p, (-3, 4, -3, 4), num=200)

    for idx, (ax, q, p) in enumerate(zip(axes.ravel(), positions, momentums)):
        ax.imshow(Z, alpha=0.9, extent=extent, cmap='bone_r', origin='upper')

        ax.quiver(q[steps, 0], q[steps, 1], p[steps, 0], p[steps, 1], headwidth=6, scale=60,
                  headlength=7, color='C1')
        ax.plot(q[:, 0], q[:, 1], '-', lw=3, color='C1')
    # plt.savefig('mixture_leapfrog.png')
    plt.show()


###

def plot3():
    np.random.seed(4)

    neg_log_p = neg_log_mvnormal(np.zeros(2), np.eye(2))
    ss, pp, mm, pl = [], [], [], [1, 2, 4]
    for path_len in pl:
        samples, positions, momentums, accepted, _ = hamiltonian_monte_carlo2(
            10, neg_log_p, np.random.randn(2), path_len=path_len, step_size=0.01)
        ss.append(samples)
        pp.append(positions)
        mm.append(momentums)

    fig, axes = plt.subplots(ncols=len(ss), figsize=(7 * len(ss), 7))

    Z, extent = neg_log_p_to_img(neg_log_p, (-3, 3, -3, 3), num=200)
    steps = slice(None, None, 20)

    for ax, samples, positions, momentums, path_len in zip(axes.ravel(), ss, pp, mm, pl):
        ax.imshow(Z, alpha=0.9, extent=extent, cmap='bone_r', origin='upper')

        for q, p in zip(positions, momentums):
            ax.quiver(q[steps, 0], q[steps, 1], p[steps, 0], p[steps, 1], headwidth=6, scale=60,
                      headlength=7, color='C1')
            ax.plot(q[:, 0], q[:, 1], '-', color='C1', lw=3)

        ax.plot(samples[:, 0], samples[:, 1], 'o', color='black', mfc='C1', ms=10)
        ax.set_title(f'Path length of {path_len}')
    # plt.savefig('normal_hmc.png')
    plt.show()


def plot4():
    plt.rcParams['text.usetex'] = True

    np.random.seed(28)  # 13 14 25

    mu1 = np.ones(2)
    cov1 = 0.5 * np.array([[1., 0.7],
                           [0.7, 1.]])
    mu2 = -mu1
    cov2 = 0.2 * np.array([[1., -0.6],
                           [-0.6, 1.]])
    mu3 = np.array([-1., 2.])
    cov3 = 0.3 * np.eye(2)

    neg_log_p = mixture(
        [neg_log_mvnormal(mu1, cov1), neg_log_mvnormal(mu2, cov2), neg_log_mvnormal(mu3, cov3)],
        [0.3, 0.3, 0.4])

    neg_log_p = lambda p: 4 * ((p ** 2).sum() ** 0.5 - 2) ** 2 + 0.1 * ((p - 2) ** 2).sum()

    ss, pp, mm, pl = [], [], [], [1, 2, 4, 8]
    for path_len in pl:
        samples, positions, momentums, accepted, _ = hamiltonian_monte_carlo2(
            10, neg_log_p, np.random.randn(2), path_len=path_len, step_size=0.01)
        ss.append(samples)
        pp.append(positions)
        mm.append(momentums)

    fig, axes = plt.subplots(ncols=2, nrows=len(ss) // 2,
                             figsize=(3 * len(ss) / 2, 3 * 2))

    Z, extent = neg_log_p_to_img(neg_log_p, (-3, 3, -3, 3), num=200)
    steps = slice(None, None, 20)

    for ax, samples, positions, momentums, path_len in zip(axes.ravel(), ss, pp, mm, pl):
        ax.imshow(Z, alpha=0.9, extent=extent, cmap='bone_r', origin='upper')

        for q, p in zip(positions, momentums):
            ax.quiver(q[steps, 0], q[steps, 1], p[steps, 0], p[steps, 1], headwidth=6, scale=60,
                      headlength=7, color='sienna')
            ax.plot(q[:, 0], q[:, 1], '-', color='sienna', lw=1)

        ax.plot(samples[:, 0], samples[:, 1], 'o', color='w', mfc='sienna', ms=5)
        ax.set_title(f'$\epsilon=0.01$, $L\epsilon={path_len}$')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
    plt.tight_layout()
    plt.savefig('hmc_plot.pdf')
    plt.show()


plot4()
