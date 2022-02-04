import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit, prange, float64, int64

# Function to calculate energy of a given configuration
#   if atom_i is specified, then the energy is calculated by updating
#   prev_energy due to atom_i moving by pos_change=[dx,dy]
@jit(nopython=True)
def potential(atoms_pos):
    energy = 0
    for i in range(N_atoms):
        pos_i = atoms_pos[i]
        for j in range(N_atoms):
            if j >= i:
                break
            pos_j = atoms_pos[j]
            r_ij = pos_i - pos_j
            # find minimum distance image
            r_ij = r_ij - box_size*np.rint(r_ij/box_size)
            dist = np.linalg.norm(r_ij)
            energy += 1/dist**12 - 1/dist**6
    return energy

# calculate energy by updating previous energy
@jit(float64(float64[:,:], int64, float64[:], float64, int64), nopython=True)
def potential_update(atoms_pos, atom_i, pos_change, prev_energy, box_size):
    N_atoms = len(atoms_pos)
    energy = prev_energy
    for i in prange(N_atoms):
        if i == atom_i:
            continue
        # old distance
        r_ij = atoms_pos[atom_i] - atoms_pos[i]
        r_ij = r_ij - box_size*np.rint(r_ij/box_size)
        dist = np.linalg.norm(r_ij)
        V_ij = 1/dist**12 - 1/dist**6
        energy = energy - V_ij
        # new distance
        r_ij = atoms_pos[atom_i] + pos_change - atoms_pos[i]
        r_ij = r_ij - box_size*np.rint(r_ij/box_size)
        dist = np.linalg.norm(r_ij)
        V_ij = 1/dist**12 - 1/dist**6
        energy = energy + V_ij
    return energy

# Calculate pair distribution function g(r)
#   atom_i - index of atom to calculate distribution around
#   r_bins - distance bins over which to calculate g(r)
def pair_distribution(atoms_pos, N_atoms, atom_i, box_size, r_bins):
    density = N_atoms / box_size**2
    n_bins = len(r_bins) - 1
    max_r = r_bins[-1]
    bin_size = max_r / n_bins
    bins = np.zeros(n_bins)
    for i in range(N_atoms):
        if i == atom_i:
            continue
        r_ij = atoms_pos[atom_i] - atoms_pos[i]
        r_ij = r_ij - box_size*np.rint(r_ij/box_size)
        dist = np.linalg.norm(r_ij)
        if dist < max_r:
            bin_i = int(dist / bin_size)
            bins[bin_i] += 1
    bin_volumes = (2*np.arange(n_bins) + 1) * np.pi
    g = np.zeros(n_bins+1)
    g[1:] = bins / (bin_volumes * density)
    return g



if __name__ == "__main__":
    np.random.seed(100)

    N_atoms = 50
    box_size = 15
    k_b = 8.617e-5
    T = 6e3

    dr = 0.4

    iterations = int(1e4)
    start_average = 0

    energies = np.zeros(iterations)

    time_start = time.perf_counter()

    # initialize a random configuaration of atoms in 2D
    atoms_pos = np.random.rand(N_atoms, 2)*box_size

    # initial energy
    energy = potential(atoms_pos)
    print("Initial energy {:.2e}".format(energy))

    rejected = 0

    r_bins = np.linspace(0, box_size/2, 15)
    gr = np.zeros(len(r_bins))
    n_pdfs = 0

    relaxed = False
    energy_p100 = energy

    # number of steps to use when calculating the adaptive step size
    adaptive_rate = 500
    rejected_ = 0

    time_start_i = time.perf_counter()
    for i in range(iterations):

        particle_i = np.random.randint(0, N_atoms)
        angle = np.random.rand()*2*np.pi
        dxy = dr * np.array([np.cos(angle), np.sin(angle)])

        new_energy = potential_update(atoms_pos, particle_i, dxy, energy, box_size)

        if new_energy > energy:
            prob = np.exp((energy - new_energy) / (k_b * T))
            if prob < np.random.rand():
                # reject move and keep old energy
                rejected += 1
                rejected_ += 1
                new_energy = energy

        # update position and energy
        if energy != new_energy:
            atoms_pos[particle_i] += dxy
            # apply periodic BC
            atoms_pos = atoms_pos - box_size * np.floor(atoms_pos / box_size)
        energy = new_energy
        energies[i] = energy

        # determine whether system has relaxed
        # change in energy over 100 snapshots should be less than 15
        if relaxed == False:
            if i % 100 == 0 and i != 0:
                if np.abs(new_energy - energy_p100) <= 15:
                    relaxed = True
                    start_average = i-50
                else:
                    energy_p100 = new_energy

        # adaptive stepsize
        if i % adaptive_rate == 0 and i != 0:
            if rejected_ / adaptive_rate > 0.7:
                dr /= 2
            elif rejected_ / adaptive_rate < 0.5 and dr < box_size / 2:
                dr *= 2
            rejected_ = 0

        if i > start_average:
            gr += pair_distribution(atoms_pos, N_atoms, np.random.randint(0,N_atoms),
                                    box_size, r_bins)
            n_pdfs += 1

    time_end_i = time.perf_counter()
    elapsed = time_end_i - time_start_i

    gr /= n_pdfs

    time_end = time.perf_counter()

    print("Average energy:", round(np.mean(energies[start_average:]), 3))
    print("Rejection rate: {:.1f}%".format(rejected*100/iterations))
    print("{:d} iterations in {:.2f} seconds, {:.2f} ms per iteration".format(
        iterations, elapsed, elapsed*1000/iterations))
    print("Total time elapsed: {:.2f} seconds".format(time_end-time_start))

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].plot(atoms_pos[:,0], atoms_pos[:,1], '.')
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("Final configuration")
    axs[1].plot(r_bins, gr)
    axs[1].set_xlabel("r")
    axs[1].set_ylabel("g(r)")
    axs[1].set_title("Pair distribution function")
    plt.suptitle("T = {:.1e}".format(T))
    plt.show()

    plt.plot(np.arange(start_average-50, iterations), energies[start_average-50:])
    plt.xlabel("MC step")
    plt.ylabel("Energy")
    plt.show()
