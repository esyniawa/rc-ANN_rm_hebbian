import os
import numpy as np
import ANNarchy as ann
import matplotlib.pyplot as plt

from .definitions import InputNeuron, ReservoirNeuron, OutputNeuron, EHLearningRule


class RCNetwork:
    def __init__(self,
                 dim_reservoir: int | tuple,
                 dim_out: int,
                 sigma: float = 0.2,
                 rho: float = 1.2,
                 phi: float = 0.5):

        self.dim_reservoir = dim_reservoir
        self.dim_out = dim_out

        self.network = RCNetwork.build_network(dim_reservoir=self.dim_reservoir,
                                               dim_out=self.dim_out, sigma=sigma, rho=rho, phi=phi)

        self.monitors = []
        self.monitors_vars = []

    @staticmethod
    def build_network(dim_reservoir: int | tuple,
                      dim_out: int,
                      sigma: float = 0.2,
                      rho: float = 1.2,
                      phi: float = 0.6):

        fb_strength = 1.0
        target_strength = 1.0

        if isinstance(dim_reservoir, tuple):
            N_res = np.prod(dim_reservoir)
        else:
            N_res = dim_reservoir

        # Target population
        target_pop = ann.Population(geometry=dim_out, neuron=InputNeuron, name='target_pop')

        # Built reservoir
        reservoir = ann.Population(geometry=dim_reservoir, neuron=ReservoirNeuron, name='reservoir_pop')
        reservoir.chaos_factor = rho
        reservoir.tau = 10.

        # output population
        output_pop = ann.Population(geometry=dim_out, neuron=OutputNeuron, name='output_pop')
        output_pop.phi = phi

        # connections
        recurrent_res = ann.Projection(pre=reservoir, post=reservoir, target='rec')
        recurrent_res.connect_fixed_probability(probability=sigma,
                                                weights=ann.Normal(mu=0,
                                                                   sigma=np.sqrt(1 / (sigma * N_res))))

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

    def add_input(self, dim_in: int, scale_input: float = 1.0, name: str = 'input_pop'):
        pop = ann.Population(geometry=dim_in, neuron=InputNeuron, name=name)
        res = self.network.get_population(name='reservoir_pop')

        proj = ann.Projection(pre=pop, post=res, name=name + '_in')
        proj.connect_all_to_all(ann.Uniform(min=-scale_input, max=scale_input))

        self.network.add([pop, proj])

    def add_output(self, dim: int, name: str, scale_fb: float | None, scale_target: float = 1.0):

        # new pops
        pop_out = ann.Population(geometry=dim, neuron=OutputNeuron, name='out_' + name)
        pop_target = ann.Population(geometry=dim, neuron=InputNeuron, name='target_' + name)

        res = self.network.get_population(name='reservoir_pop')

        # new projs
        proj_out = ann.Projection(pre=res, post=pop_out, name='con_' + name + '_out')
        proj_out.connect_all_to_all(0.0)

        proj_target = ann.Projection(pre=pop_target, post=pop_out, name='con_' + name + '_target')
        proj_target.connect_one_to_one(scale_target)

        if scale_fb is not None:
            proj_fb = ann.Projection(pre=pop_out, post=res, name='con_' + name + '_fb')
            proj_fb.connect_all_to_all(ann.Uniform(min=-scale_fb, max=scale_fb))

            self.network.add([pop_out, pop_target, proj_out, proj_target, proj_fb])
        else:
            self.network.add([pop_out, pop_target, proj_out, proj_target])

    def compile_network(self, folder: str):
        self.network.compile(directory=folder)

    def run_target(self, data_target: np.ndarray, period: float = 1.,
                   training: bool = True, sim_time: float | None = None):

        if training:
            self.network.enable_learning()
        else:
            self.network.disable_learning()
            self.network.get_population(name='output_pop').phi = 0.0

        @ann.every(period=period, net_id=self.network.id)
        def set_inputs(n):
            # Set inputs to the network
            self.network.get_population(name='target_pop').baseline = data_target[n]

        ann.enable_callbacks(net_id=self.network.id)
        if sim_time is None:
            t_data_in = data_target.shape[0]
            self.network.simulate(t_data_in * period)
        else:
            self.network.simulate(sim_time * period)

    @staticmethod
    def make_target(dim_out: int, n_trials: int, seed: None | int = None):

        # random period time
        T = np.random.RandomState(seed).uniform(1000, 2000)
        x = np.arange(0, n_trials * T)

        y = np.zeros((len(x), dim_out))

        for out in range(dim_out):

            a1 = np.random.RandomState(seed).normal(loc=0, scale=1)
            a2 = np.random.RandomState(seed).normal(loc=0, scale=1)
            a3 = np.random.RandomState(seed).normal(loc=0, scale=0.5)

            y[:, out] = a1 * np.sin(2 * np.pi * x / T) + a2 * np.sin(4 * np.pi * x / T) + a3 * np.sin(6 * np.pi * x / T)

        return y, T

    def init_monitors(self, pop_names: list[str], var_names: list[str] | None = None, sample_rate: float = 2.0):
        if var_names is None:
            var_names = ['r'] * len(pop_names)

        for pop_name, var_name in zip(pop_names, var_names):
            pop = self.network.get_population(name=pop_name)
            self.monitors.append(ann.Monitor(pop, variables=var_name, start=True, period=sample_rate))

        self.network.add(self.monitors)

    def get_monitors(self, delete_monitors: bool = True, reshape: bool = True) -> dict:
        """
        Gets all initialized monitors and returns them in a dict
        :param delete_monitors:
        :param reshape:
        :return:
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

    def plot_rates(self, plot_order: tuple[int, int],
                   plot_types: tuple,
                   fig_size: tuple[float, float] | list[float, float] = (5, 5),
                   save_name: str = None) -> None:

        """
        Plots 2D populations rates.
        :param plot_type: can be 'Plot' or 'Matrix'
        :param plot_order:
        :param fig_size:
        :param save_name:
        :return:
        """

        from .utils import ceil, reshape_array

        ncols, nrows = plot_order
        results = self.get_monitors(delete_monitors=False, reshape=True)

        fig = plt.figure(figsize=fig_size)
        for i, key_pop in enumerate(results):
            plot_type = plot_types[i]

            for j, key_var in enumerate(results[key_pop]):
                plt.subplot(nrows, ncols, i + j + 1)
                if plot_type == 'Plot':
                    if results[key_pop][key_var].ndim > 2:
                        results[key_pop][key_var] = reshape_array(results[key_pop][key_var])

                    plt.plot(results[key_pop][key_var])
                    plt.ylabel(key_var)
                    plt.xlabel('t', loc='right')

                elif plot_type == 'Matrix':
                    if results[key_pop][key_var].ndim > 3:
                        results[key_pop][key_var] = reshape_array(results[key_pop][key_var], dim=3)

                    res_max = np.amax(abs(results[key_pop][key_var]))
                    img = plt.contourf(results[key_pop][key_var].T, cmap='RdBu', vmin=-res_max, vmax=res_max)
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
