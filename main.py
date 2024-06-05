import numpy as np

from network.model import RCNetwork


def closed_loop_network(dim_reservoir: int | tuple,
                        dim_out: int,
                        n_periods_train: int,
                        n_periods_test: float,
                        plot_name: str | None = None):

    # make input
    target, period = RCNetwork.make_dynamic_target(dim_out=dim_out, n_trials=n_periods_train)

    # make network
    RC = RCNetwork(dim_reservoir=dim_reservoir, dim_out=dim_out, rho=1.5)
    RC.compile_network(folder="annarchy/closed_loop_net/")

    # init monitors
    monitoring_rates = [pop.name for pop in RC.get_all_populations()]
    RC.init_monitors(pop_names=monitoring_rates)

    # simulate
    RC.run_target(data_target=target)
    RC.run_target(data_target=target, training=False, sim_time=n_periods_test * period)

    RC.save_monitors(folder='results/closed_loop_run/', delete_monitors=False)
    RC.plot_rates(
        plot_order=(1, len(monitoring_rates)),
        plot_types=('Plot', 'Matrix', 'Plot'),
        save_name=plot_name,
        t_init=int((n_periods_train-3) * period),
        fig_size=(16, 14)
    )


def open_loop_network(dim_reservoir: int | tuple,
                      n_periods_train: int,
                      n_periods_test: int,
                      plot_name: str | None = None):

    from network.utils import memory_out

    # make inputs
    train_input_memory, train_memory_trace, train_T = RCNetwork.make_memory_trace(n_changes=n_periods_train,
                                                                                  heavyside_offset=-5,
                                                                                  norm=False)

    train_input_rng, train_nonlinear_trace = RCNetwork.make_random_walk_clip(dim_out=2, T=train_T, bounderies=1.5)
    train_memory_out = memory_out(memory_trace=train_memory_trace, rng_trace=train_input_rng)

    test_input_memory, test_memory_trace, test_T = RCNetwork.make_memory_trace(n_changes=n_periods_test,
                                                                               heavyside_offset=-5,
                                                                               norm=False)

    test_input_rng, test_nonlinear_trace = RCNetwork.make_random_walk_clip(dim_out=2, T=test_T, bounderies=1.5)
    test_memory_out = memory_out(memory_trace=test_memory_trace, rng_trace=test_input_rng)

    # concatenate arrays
    input_training = np.column_stack((train_input_memory, train_input_rng))
    target_non_fb_training = np.column_stack((train_nonlinear_trace, train_memory_out))

    input_test = np.column_stack((test_input_memory, test_input_rng))
    target_non_fb_test = np.column_stack((test_nonlinear_trace, test_memory_out))

    # make network
    RC = RCNetwork(dim_reservoir=dim_reservoir, dim_out=1, rho=1.2)

    # make input populations
    RC.add_input(dim_in=4, name='in_memory_task')

    # make output populations
    RC.add_output(dim=2, name='memory_task', scale_fb=None)

    # build network
    RC.compile_network(folder="annarchy/open_loop_net/")

    # init monitors
    monitoring_rates = [pop.name for pop in RC.get_all_populations()]
    RC.init_monitors(pop_names=monitoring_rates)

    # training
    RC.run_targets_with_inputs(data_in=input_training,
                               data_target_open=target_non_fb_training,
                               data_target_closed=train_memory_trace)

    # testing
    RC.run_targets_with_inputs(data_in=input_test,
                               data_target_open=target_non_fb_test,
                               data_target_closed=test_memory_trace,
                               training=False)

    RC.save_monitors(folder='results/open_loop_run/', delete_monitors=False)
    RC.plot_rates(
        plot_order=(1, len(monitoring_rates)),
        plot_types=('Plot', 'Matrix', 'Plot', 'Plot', 'Plot', 'Plot'),
        save_name=plot_name,
        t_init=-4*test_T,
        fig_size=(16, 14)
    )


if __name__ == '__main__':
    # task 1
    closed_loop_network(1000, 2, 50, n_periods_test=5)
    # task 3
    open_loop_network(1000, 50, 5)
