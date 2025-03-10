# exploration-mechanisms
Experiments on mechanisms to enhance exploration in metaheuristics

Specifically, several mechanisms are incorporated into Particle Swarm Optimization to enhance exploration and prevent
premature convergence.
The following algorithms are available for tuning and comparison:

1) PSO
2) SHADE
3) PSO with Random Restarts
4) PSO-NPGM (New Population Generation Mechanism)
5) PSO-GPGM (Gbest-guided Population Generation Mechanism)
6) PSO-SRM (Solution Replacement Mechanism)
7) PSO-PDM (Population Dispersion Mechanism)

The algorithm implementations utilise the MAHF framework and can be found in `src/algorithms`.


### Hyperparameter tuning

The setup for tuning using `irace` can be found in `src/bin/irace_tuning` for the algorithmic configuration
and `src/tuning` for the `irace` configuration.

To execute, open a terminal, navigate to the respective subfolder of the algorithmic variant (e.g. `../tuning/pso`)
and run `irace` (we use the command line option of irace; for more details on that, take a look at the official documentation).


### Mechanism Comparison

The setup for the comparison of the different strategies with optimized parameter settings can be found in
`src/bin/exploration_experiment`.

The experiments can be run using

cargo run --release experiment_name --function f --dimension --d


### Full Examples

If you are interested in details of the results of additional configurations, you can use and adapt the code
in `src/bin/full_examples`, which evaluates hyperparameter settings individually.