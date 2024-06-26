from typing import Optional, Tuple
import os
import numpy as np
import ANNarchy as ann
import matplotlib.pyplot as plt

from .definitions import InputNeuron, InputNeuron_dynamic, ReservoirNeuron, OutputNeuron, EHLearningRule


class RCNetwork:
    def __init__(self,
                 dim_reservoir: int | tuple,
                 dim_out: int,
                 sigma: float = 0.2,
                 rho: float = 1.2,
                 phi: float = 0.5,
                 tau: float = 10.):

        # dimensions
        self.dim_reservoir = dim_reservoir
        self.dim_out = dim_out

        # parameters
        self.sigma = sigma
        self.rho = rho
        self.phi = phi
        self.tau = tau

        # init network
        self.output_pops = []
        self.input_pops = []
        self.target_pops = []

        self.network = self.build_network()

        # monitors for recording
        self.monitors = []
        self.sample_rate = 2.0

    def build_network(self):
        """
        Builds the reservoir computing network with specified parameters and connections.

        :return: ann.Network: The constructed ANNarchy network object.
        """

        fb_strength = 1.0
        target_strength = 1.0

        if isinstance(self.dim_reservoir, tuple):
            N_res = np.prod(self.dim_reservoir)
        else:
            N_res = self.dim_reservoir

        # Target population
        target_pop = ann.Population(geometry=self.dim_out, neuron=InputNeuron, name='target_pop')
        self.target_pops.append(target_pop.name)

        # Built reservoir
        reservoir = ann.Population(geometry=self.dim_reservoir, neuron=ReservoirNeuron, name='reservoir_pop')
        reservoir.chaos_factor = self.rho
        reservoir.tau = self.tau

        # output population
        output_pop = ann.Population(geometry=self.dim_out, neuron=OutputNeuron, name='output_pop')
        output_pop.phi = self.phi
        self.output_pops.append(output_pop.name)

        # connections
        recurrent_res = ann.Projection(pre=reservoir, post=reservoir, target='rec')
        recurrent_res.connect_fixed_probability(probability=self.sigma,
                                                weights=ann.Normal(mu=0,
                                                                   sigma=np.sqrt(1 / (self.sigma * N_res))))

        # reservoir -> output
        res_output = ann.Projection(pre=reservoir, post=output_pop,
                                    target='in',
                                    synapse=EHLearningRule,
                                    name='res_out_con')
        res_output.connect_all_to_all(weights=0.0)  # set it to a very small value

        # target -> output
        proj_target_output = ann.Projection(pre=target_pop, post=output_pop, target='tar')
        proj_target_output.connect_one_to_one(weights=target_strength)

        # feedback output -> reservoir
        output_reservoir_proj = ann.Projection(pre=output_pop, post=reservoir, target='fb')
        output_reservoir_proj.connect_all_to_all(weights=ann.Uniform(-fb_strength, fb_strength))

        network = ann.Network(everything=True)

        return network

    def add_input(self, dim_in: int,
                  scale_input: float = 1.0,
                  name: str = 'input_pop',
                  neuron_model: ann.Neuron = InputNeuron):
        """
        Adds an input population to the reservoir computing network.

        :param dim_in: The dimensionality of the input population.
        :param scale_input: The scaling factor for the input weights. Default is 1.0.
        :param name: The name of the input population. Default is 'input_pop'.
        :param neuron_model: The neuron model to use for the input population. Default is InputNeuron.
        """

        # new pop
        pop = ann.Population(geometry=dim_in, neuron=neuron_model, name=name)
        self.input_pops.append(name)

        res = self.network.get_population(name='reservoir_pop')

        proj = ann.Projection(pre=pop, post=res, target='in', name=name + '_in')
        proj.connect_all_to_all(ann.Uniform(min=-scale_input, max=scale_input))

        self.network.add([pop, proj])

    def add_output(self, dim: int, name: str, scale_fb: float | None, scale_target: float = 1.0):
        """
        Adds an output population to the reservoir computing network.

        :param dim: The dimensionality of the output population.
        :param name: The name of the output population.
        :param scale_fb: The scaling factor for the feedback weights. If None, no feedback projection is created.
        :param scale_target: The scaling factor for the target weights. Default is 1.0.
        """

        # new pops
        pop_out = ann.Population(geometry=dim, neuron=OutputNeuron, name='out_' + name)
        pop_target = ann.Population(geometry=dim, neuron=InputNeuron, name='target_' + name)
        self.output_pops.append(pop_out.name)
        self.target_pops.append(pop_target.name)

        res = self.network.get_population(name='reservoir_pop')

        # new projs
        proj_out = ann.Projection(pre=res, post=pop_out,
                                  target='in',
                                  synapse=EHLearningRule,
                                  name='con_' + name + '_out')
        proj_out.connect_all_to_all(0.0)

        proj_target = ann.Projection(pre=pop_target, post=pop_out, target='tar', name='con_' + name + '_target')
        proj_target.connect_one_to_one(scale_target)

        if scale_fb is not None:
            proj_fb = ann.Projection(pre=pop_out, post=res, target='fb', name='con_' + name + '_fb')
            proj_fb.connect_all_to_all(ann.Uniform(min=-scale_fb, max=scale_fb))

            self.network.add([pop_out, pop_target, proj_out, proj_target, proj_fb])
        else:
            self.network.add([pop_out, pop_target, proj_out, proj_target])

    def compile_network(self, folder: str):
        self.network.compile(directory=folder)

    def get_all_projections(self):
        return self.network.get_projections()

    def get_all_populations(self):
        return self.network.get_populations()

    def run_target(self, data_target: np.ndarray, period: float = 1.,
                   training: bool = True, sim_time: float | None = None):

        if training:
            self.network.enable_learning()
            self.network.get_population(name=self.output_pops[0]).test = 0
        else:
            self.network.disable_learning()
            self.network.get_population(name=self.output_pops[0]).test = 1

        @ann.every(period=period, net_id=self.network.id)
        def set_inputs(n):
            # Set inputs to the network
            self.network.get_population(name=self.target_pops[0]).baseline = data_target[n]

        ann.enable_callbacks(net_id=self.network.id)
        if sim_time is None:
            t_data_in = data_target.shape[0]
            self.network.simulate(t_data_in * period)
        else:
            self.network.simulate(sim_time * period)

    def run_targets_with_inputs(self,
                                data_target_closed: np.ndarray,
                                data_target_open: np.ndarray,
                                data_in: np.ndarray,
                                period: float = 1.,
                                training: bool = True,
                                sim_time: float | None = None):

        if training:
            self.network.enable_learning()
            for out_pop in self.output_pops:
                self.network.get_population(name=out_pop).test = 0
        else:
            self.network.disable_learning()
            for out_pop in self.output_pops:
                self.network.get_population(name=out_pop).test = 1

        @ann.every(period=period, net_id=self.network.id)
        def set_inputs(n):
            # TODO: This implementation isn't optimal. You could track the input population and iterate over them.
            #  But I didn't want to implement a second for-loop, therefore this implementation
            self.network.get_population(name=self.target_pops[0]).baseline = data_target_closed[n]
            self.network.get_population(name=self.input_pops[0]).baseline = data_in[n]
            self.network.get_population(name=self.target_pops[1]).baseline = data_target_open[n]

        ann.enable_callbacks(net_id=self.network.id)
        if sim_time is None:
            t_data_in = data_target_closed.shape[0]
            self.network.simulate(t_data_in * period)
        else:
            self.network.simulate(sim_time * period)

    @staticmethod
    def make_dynamic_target(dim_out: int, n_trials: int, seed: Optional[int] = None):
        """
        Generates a dynamic target signal for the reservoir computing network.

        :param dim_out: The dimensionality of the output signal.
        :param n_trials: The number of trials for which the signal is generated.
        :param seed: The seed for the random number generator. Default is None.

        :return: A tuple containing the generated dynamic target signal (numpy array) and the period time (float).
        """

        # random period time
        T = np.random.RandomState(seed).uniform(1500, 2000)
        x = np.arange(0, n_trials * T)

        y = np.zeros((len(x), dim_out))

        for out in range(dim_out):

            a1 = np.random.RandomState(seed).normal(loc=0, scale=1)
            a2 = np.random.RandomState(seed).normal(loc=0, scale=1)
            a3 = np.random.RandomState(seed).normal(loc=0, scale=0.5)

            y[:, out] = a1 * np.sin(2 * np.pi * x / T) + a2 * np.sin(4 * np.pi * x / T) + a3 * np.sin(6 * np.pi * x / T)

        return y, T

    @staticmethod
    def make_memory_trace(n_changes: int,
                          heavyside_width: int = 100,
                          heavyside_offset: int = 0,
                          min_length: int = 300,
                          max_length: int = 1200,
                          smoothing_window: int = 100,
                          scale_output: float | None = None,
                          seed: Optional[int] = None):

        """
        Generates a memory trace signal for the reservoir computing network.

        :param n_changes: The number of changes in the memory trace signal.
        :param heavyside_width: The width of the heavyside function for each change. Default is 100.
        :param heavyside_offset: The offset of the heavyside input function for each change. Default is 0.
        :param min_length: The minimum time steps of each change. Default is 300.
        :param max_length: The maximum time steps of each change. Default is 1200.
        :param smoothing_window: The window size for exponential smoothing of the memory trace and input trace. Default is 100.
        :param scale_output: The scaling factor for the output. If None, the output is not scaled. Default is None.
        :param seed: The seed for the random number generator. Default is None.

        :return: A tuple containing the input trace (numpy array), memory trace (numpy array), and the length of the input trace.
        """

        from .utils import exponential_smoothing

        # the memory trace always begins with an off state and should end with the off state, therefore
        # n_changes must be odd
        if n_changes % 2 == 0:
            n_changes += 1

        changes = np.random.RandomState(seed).randint(low=min_length, high=max_length, size=n_changes)
        T = np.sum(changes)
        changes = np.cumsum(changes)

        memory_trace = np.zeros(T)
        input_trace = np.zeros((T, 2))

        for i in range(len(changes) - 1):
            memory_trace[int(changes[i]):int(changes[i+1])] = (i+1) % 2
            start, end = int(changes[i] + heavyside_offset), int(changes[i] + heavyside_width + heavyside_offset)
            input_trace[start:end, i % 2] = 1

        # smooth input and target output
        memory_trace = exponential_smoothing(memory_trace, windows_size=smoothing_window) - .5
        input_trace = exponential_smoothing(input_trace, windows_size=smoothing_window)

        if scale_output is None:
            return input_trace, memory_trace, input_trace.shape[0]
        else:
            return scale_output * input_trace, scale_output * memory_trace, input_trace.shape[0]

    @staticmethod
    def make_random_walk(dim_out: int, T: int,
                         start_points: float | np.ndarray = 0.0,
                         mu: float = 0.0,
                         sigma: float = 0.02,
                         seed: Optional[int] = None):

        # generate random gradients
        rng = np.random.default_rng(seed).normal(loc=mu, scale=sigma, size=(T-1, dim_out))
        # if single value fill a array with the fitting dimensions
        if isinstance(start_points, float):
            start_points = np.repeat(start_points, dim_out)

        trace = np.cumsum(np.row_stack((start_points, rng)), axis=0)

        return trace, np.mean(trace, axis=1)

    @staticmethod
    def make_random_walk_clip(dim_out: int, T: int, bounderies: float,
                              start_points: float | np.ndarray = 0.0,
                              mu: float = 0.0,
                              sigma: float = 0.05,
                              seed: Optional[int] = None):

        # Initialize random number generator
        rng = np.random.default_rng(seed)

        # Generate random gradients
        random_steps = rng.normal(loc=mu, scale=sigma, size=(T-1, dim_out))

        # If start_points is a single value, create an array with the same value repeated
        if isinstance(start_points, float):
            start_points = np.full(dim_out, start_points)

        # Initialize the trace with the start points
        trace = np.zeros((T, dim_out))
        trace[0] = start_points

        # Compute the random walk
        for t in range(1, T):
            trace[t] = trace[t-1] + random_steps[t-1]
            # Clip the values to the specified boundaries
            trace[t] = np.clip(trace[t], -bounderies, bounderies)

        return trace, np.mean(trace, axis=1)

    def init_monitors(self, pop_names: list[str], var_names: list[str] | None = None, sample_rate: float = 2.0):
        self.sample_rate = sample_rate

        if var_names is None:
            var_names = ['r'] * len(pop_names)

        for pop_name, var_name in zip(pop_names, var_names):
            pop = self.network.get_population(name=pop_name)
            self.monitors.append(ann.Monitor(pop, variables=var_name, start=True, period=self.sample_rate))

        self.network.add(self.monitors)

    def get_monitors(self, delete_monitors: bool = True, reshape: bool = True) -> dict:
        """
        Gets all initialized monitors and returns them in a dict
        :param delete_monitors:
        :param reshape:
        :return: Monitor dict.
        """

        res = {}
        for monitor in self.monitors:
            res[monitor.object.name] = self.network.get(monitor).get(keep=not delete_monitors, reshape=reshape)

        return res

    def save_monitors(self, folder: str, delete_monitors: bool = True, reshape: bool = True):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for monitor in self.monitors:

            rec = self.network.get(monitor)
            np.save(folder + monitor.object.name, rec.get(keep=not delete_monitors, reshape=reshape))

    def plot_rates(self,
                   plot_order: tuple[int, int],
                   plot_types: tuple | None,
                   t_init: int = 0,
                   fig_size: tuple[float, float] | list[float, float] = (5, 5),
                   save_name: str = None) -> None:

        """
        Plots 2D populations rates.
        :param plot_types: A tuple containing plot types for each population rate. Can be 'Plot', 'Matrix', or None
        :param plot_order: tuple specifying the number of columns and rows for the subplots grid.
        :param t_init: The initial time step to start plotting from. Default is 0.
        :param fig_size: The size of the figure. Default is (5, 5).
        :param save_name: The name of the file to save the plot. If None, the plot will be displayed instead.
        """

        from .utils import ceil, reshape_array

        ncols, nrows = plot_order
        results = self.get_monitors(delete_monitors=False, reshape=True)
        t_init = int(t_init / self.sample_rate)

        fig = plt.figure(figsize=fig_size)
        for i, key_pop in enumerate(results):
            plot_type = plot_types[i]

            for j, key_var in enumerate(results[key_pop]):
                plt.subplot(nrows, ncols, i + j + 1)
                if plot_type == 'Plot':
                    if results[key_pop][key_var].ndim > 2:
                        results[key_pop][key_var] = reshape_array(results[key_pop][key_var])

                    plt.plot(results[key_pop][key_var][t_init:])
                    plt.ylabel(key_var)
                    plt.xlabel('t', loc='right')

                elif plot_type == 'Matrix':
                    if results[key_pop][key_var].ndim > 3:
                        results[key_pop][key_var] = reshape_array(results[key_pop][key_var], dim=3)

                    res_max = np.amax(abs(results[key_pop][key_var]))
                    img = plt.contourf(results[key_pop][key_var][t_init:].T, cmap='RdBu', vmin=-res_max, vmax=res_max)
                    plt.colorbar(img, label=key_var, orientation='horizontal')
                    plt.xlabel('t', loc='right')

                plt.title(key_pop, loc='left')

        if save_name is None:
            plt.show()
        else:
            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(save_name)
            plt.close(fig)
