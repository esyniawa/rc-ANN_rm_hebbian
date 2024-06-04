import ANNarchy as ann

ReservoirNeuron = ann.Neuron(
    parameters="""
    tau = 50.0 : population
    phi = 0.05 : population
    chaos_factor = 1.5 : population
    """,
    equations="""
    noise = phi * Uniform(-1.0, 1.0)
    x += dt*(chaos_factor * sum(rec) + sum(fb) + sum(in) - x)/tau
    r = tanh(x) + noise
    """
)

OutputNeuron = ann.Neuron(
    parameters="""
    alpha_r = 0.8 : population
    alpha_p = 0.8 : population
    baseline = 0.0 : population
    phi = 0.5 : population
    test = 0 : population, bool
    """,
    equations="""
    # input from reservoir (z)
    noise = phi * Uniform(-1.0, 1.0)
    r_in = sum(in) + baseline
    r = if (test == 0):
            r_in + noise
        else:
            r_in

    r_mean = alpha_r * r_mean + (1 - alpha_r) * r

    # performance
    p = - power(r - sum(tar), 2)
    p_mean = alpha_p * p_mean + (1 - alpha_p) * p

    # modulatory signal
    m = if (p > p_mean):
            1.0
        else:
            0.0
    """
)

InputNeuron = ann.Neuron(
    parameters="""
    baseline = 0.0
    phi = 0.0 : population
    """,
    equations="""
    r = baseline + phi * Uniform(-1.0,1.0)
    """
)

TestTargetNeuron = ann.Neuron(
    parameters="""
    w1 = 0.6 : population
    w2 = 0.7 : population
    w3 = -0.2 : population

    f1 = 1/1000 : population
    f2 = 1/500 : population
    f3 = 1/500 : population

    phase_shift = -20.
    """,
    equations="""
    r = w1 * sin(2 * pi * f1 * t) + w2 * sin(2 * pi * f2 * t) + w3 * sin(2 * pi * f3 * t)
    """
)

EHLearningRule = ann.Synapse(
    parameters="""
        eta_init = 0.0005
        decay = 10000.
        """,
    equations="""
        w_old = w
        learning_rate = eta_init / (1 + t/decay)
        
        delta_w = learning_rate * (post.r - post.r_mean) * post.m * pre.r
        w += delta_w
        
        error = w * pre.r - w_old
        """
)