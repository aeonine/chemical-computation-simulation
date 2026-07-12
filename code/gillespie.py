"""Implementation of D. Gillespie's Direct Method

Gillespie, D. A general method for numerically simulating the
stochastic time evolution of coupled chemical reactions. J Comp
Phys 22(4):403-434 (1976)
"""
from random import random
import numpy as np

def get_propensity(reaction, state):
    """Calculate the propensity of a given reaction for a given state

    Parameters
    ----------
    reaction: a three-tuple of educts, products and stochastic rate
              constant. Educts and products are themselves tuples
              of indices into the state

    state:    a list of integer molecule counts

    Returns
    -------
    float: The propensity of the given reaction as defined by the
           stochastic simulation algorithm.
    """
    educts, products, c = reaction

    educts = sorted(educts)

    # reaction of type  -> X
    if len(educts) == 0:
        return c

    # reaction of type A -> X
    elif len(educts) == 1:
        idx = educts[0]
        return c * state[idx]

    # reaction of type A + B -> X
    elif len(educts) == 2 and educts[0] != educts[1]:
        idx1, idx2 = educts
        return c * state[idx1] * state[idx2]

    # reaction of type A + A -> X
    elif len(educts) == 2 and educts[0] == educts[1]:
        idx = educts[0]
        return 1/2 * c * state[idx] * (state[idx]-1)

    # reaction of type A + B + C -> X
    elif len(educts) == 3 and educts[0] != educts[1] and educts[1] != educts[2]:
        idx1, idx2, idx3 = educts
        return c * state[idx1] * state[idx2] * state[idx3]

    # reaction of type A + A + B -> X
    elif len(educts) == 3 and educts[0] == educts[1] and educts[1] != educts[2]:        
        idx1, idx2, idx3 = educts
        return 1/2 * c * state[idx1] * (state[idx1]-1) * state[idx3]

    # reaction of type A + B + B -> X
    elif len(educts) == 3 and educts[0] != educts[1] and educts[1] == educts[2]:        
        idx1, idx2, idx3 = educts
        return 1/2 * c * state[idx1] * state[idx2] * (state[idx2]-1)

    # reaction of type A + A + A -> X
    elif len(educts) == 3 and educts[0] == educts[1] and educts[1] == educts[2]:        
        idx1, idx2, idx3 = educts
        return 1/6 * c * state[idx1] * (state[idx1]-1) * (state[idx1]-2)

    raise ValueError("Reactions of order four or more are not supported.")


def gillespie_step(reactions, state, tmax, t=0):
    """Perform a single step of the stochastic simulation algorithm

    This generator function yields successive time-state tuples
    along a random sample of the stochastic process defined by the
    given reactions starting a the given state at time t. If the
    random step would exceed tmax, the state is yielded unaltered
    and the time is increamented to tmax.

    Parameters
    ----------
    reactions: a sequence of reaction tuples. Each tuple consists
               of educts, products and a stochastic rate constant
               (float). Educts and products are tuples of species
               indices into the chemical state vector.

    state:     a sequence of integer denoting molecule counts of
               each chemical species

    t, tmax:   the initial and end time as floats
    """
    # we make a copy to not modify the initial state
    while t < tmax:
        # calculate propensities
        propensities = [get_propensity(r, state) for r in reactions]
    
        # compute total propensity
        a_total = sum(propensities)

        if a_total == 0:
            # all reactions are completed
            t = tmax
            continue

        # pick random time lag
        dt = -1/a_total * np.log(random())
    
        # increase timestep
        t += dt
    
        if t <= tmax:
            # if next reaction occurs before tmax

            # pick random reaction proportional to propensity (roulette wheel selection)
            choice = a_total*random()
            for idx, a in enumerate(propensities):
                choice -= a
                if choice < 0:
                    break
    
            # perform selected reaction
            educts, products, _ = reactions[idx]
            for species in educts:
                state[species] -= 1
            for species in products:
                state[species] += 1
        else:
            # otherwise correct time
            t = tmax
    
        yield (t, state)

    yield (t, state)
    

def random_sample(reactions, state, timepoints):
    """Sample a stochastic trajectory at fixed timepoints

    For the given reaction system and state, a random sample
    is produced using Gillespie's direct method. System states
    are reported for all provided timepoints.

    Parameters
    ----------
    reactions:  a sequence of reaction tuples. Each tuple consists
                of educts, products and a stochastic rate constant
                (float). Educts and products are tuples of species
                indices into the chemical state vector.

    state:      a sequence of integer denoting molecule counts of
                each chemical species

    timepoints: a sequence of (float) timepoints for which the
                system state is recorded

    Returns
    -------
    A 2D numpy array of shape (len(timepoints), len(state)) with
    the results of the random sampling process.
    """
    # prepare results
    result = np.zeros((len(timepoints), len(state)), dtype=float)

    # for each requested timepoint
    t = timepoints[0]
    for idx, t_report in enumerate(timepoints):
        # advance the stochastic sampler to that time
        for t, state in gillespie_step(reactions, state, t_report, t):
            pass
        # and add the curret state to the results
        result[idx] = state
    return result


if __name__ == '__main__':
    # Reactions are given as educts, products, stoch. rate constant tuples
    # Educts and products are, in turn, tuples of indices into the state vector.
    reactions = [
        ((0, 1), (3,), 0.01),  # A + B -> D; k1 = 1
        ((1, 2), (4,), 0.05),  # B + C -> E; k2 = 5
    ]

    # Chemical state vector [A, B, C, D, E]
    initial = [200, 100, 100, 0, 0]

    # Sample a random stochastic trajectory
    for t, state in gillespie_step(reactions, list(initial), 10):
        ... # this section is executed during each iteration of the algorithm

    # Generate an entire stochastic trajectory
    times = np.linspace(0, 10, 101)
    trajectory = random_sample(reactions, list(initial), times)
    print(trajectory)
