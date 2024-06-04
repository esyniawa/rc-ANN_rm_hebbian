from network.model import RCNetwork


def closed_loop_network(dim_reservoir: int | tuple,
                        dim_out: int,
                        n_periods_train: int,
                        n_periods_test: float,
                        plot_name: str | None = None):

    # make input
    target, period = RCNetwork.make_target(dim_out=dim_out, n_trials=n_periods_train)

    # make network
    net = RCNetwork(dim_reservoir=dim_reservoir, dim_out=dim_out)
    net.compile_network(folder="annarchy_net/")

    # init monitors
    monitoring = ['reservoir_pop', 'output_pop', 'target_pop']
    net.init_monitors(pop_names=monitoring)

    # simulate
    net.run_target(data_target=target)
    net.run_target(data_target=target, training=False, sim_time=n_periods_test * period)

    net.save_monitors(folder='results/', delete_monitors=False)
    net.plot_rates(
        plot_order=(1, 3),
        plot_types=('Matrix', 'Plot', 'Plot'),
        save_name=plot_name
    )


if __name__ == '__main__':
    closed_loop_network(600, 2, 8, n_periods_test=2)