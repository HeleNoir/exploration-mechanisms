#[path = "../algorithms/mod.rs"]
mod algorithms;

use mahf::{prelude::*, configuration::Configuration, Random,
           lens::common::{BestObjectiveValueLens},
           components::measures::{diversity::{NormalizedDiversityLens, MinimumIndividualDistance}, },
};
use mahf_coco::{Instance, AcceleratedEvaluator, Suite, Context, Options, backends::C, Name::Bbob};

use std::{
    fs::{self},
    path::PathBuf,
    sync::{Arc},
};
use once_cell::sync::Lazy;
use clap::Parser;
use itertools::iproduct;
use mahf::prelude::common::Evaluations;
use mahf::problems::LimitedVectorProblem;
use rayon::prelude::*;
use crate::algorithms::pso_mine_explosion::mine_explosion_pso;

static CONTEXT: Lazy<Context<C>> = Lazy::new(Context::default);

#[derive(Parser)]
#[clap(version, about)]
struct Args {
    /// Number of BBOB function
    #[arg(long, default_value_t = 1)]
    function: usize,

    /// Dimensions of BBOB function
    #[arg(long, default_value_t = 2)]
    dimensions: usize,

    /// Population size of algorithm
    #[arg(long, default_value_t = 10)]
    population_size: u32,

    /// Inertia weight of PSO
    #[arg(long, default_value_t = 0.9)]
    inertia_weight: f64,

    /// C1 of PSO
    #[arg(long, default_value_t = 0.5)]
    c1: f64,

    /// C2 of PSO
    #[arg(long, default_value_t = 0.5)]
    c2: f64,

    /// Population size of exploration mechanism
    #[arg(long, default_value_t = 5)]
    new_pop: u32,
}


fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let pop_size = args.population_size;
    let functions = args.function;
    let dimensions: usize = args.dimensions;
    let inertia_weight: f64 = args.inertia_weight;
    let c1: f64 = args.c1;
    let c2: f64 = args.c2;
    let new_pop: u32 = args.new_pop;

    let folder = format!("data/mine_explosion_PSO/d{:?}_p{:?}", dimensions, pop_size);

    // set number of runs per instance
    // TODO set correctly after testing
    let runs: [usize; 5] = (1..=5).collect::<Vec<_>>()
        .try_into().expect("wrong size iterator");
    // set number of evaluations
    let evaluations: u32 = (10000 * dimensions) as u32;

    // define exploration intervals and remaining algorithmic parameters
    let restart_interval_evaluations = [evaluations as f64 * 0.1, evaluations as f64 * 0.2];
    let restart_interval_exploration = [0.05, 0.1, 0.2];
    let restarts_evaluations = vec!["evaluations"];
    let restarts_exploration = vec!["exploration"];
    let replacements = ["best", "worst", "random"];
    let center_options = ["random_new", "best", "random_solution"];

    let mut configs_eval_restart: Vec<_> = iproduct!(restarts_evaluations, restart_interval_evaluations, replacements, center_options).collect();
    let mut configs: Vec<_> = iproduct!(restarts_exploration, restart_interval_exploration, replacements, center_options).collect();
    configs.append(&mut configs_eval_restart);

    // set the benchmark problems
    let instance_indices = 1..6;
    let index: Vec<usize> = instance_indices.clone().collect();

    let n = runs.len() as u64;
    let m = index.len() as u64;

    let seeds: Vec<Vec<u64>> = (0..n)
        .map(|i| ((i * m + 1)..((i + 1) * m + 1)).collect())
        .collect();

    let options = Options::new()
        .with_dimensions([dimensions])
        .with_function_indices([functions])
        .with_instance_indices(instance_indices);
    let mut suite = Suite::with_options(Bbob, None, Some(&options)).unwrap();

    let mut problems = Vec::new();
    let mut evaluators = Vec::new();

    while let Some(instance) = suite.next() {
        let evaluator = AcceleratedEvaluator::new(&CONTEXT, &mut suite, &instance);
        problems.push(instance);
        evaluators.push(evaluator);
    }

    runs.into_par_iter()
        .zip(std::iter::repeat(evaluators).take(runs.len()).collect::<Vec<_>>())
        .for_each(|(run, evaluator)| {

            for config in &configs {

                for (i, (instance, eval)) in problems.iter().zip(evaluator.iter()).enumerate() {
                    let evaluator = eval.clone();

                    let seed = seeds[run-1][i];

                    let bounds = instance.domain();
                    let lower = bounds[0].start.clone();
                    let upper = bounds[0].end.clone();
                    let v_max = (upper - lower) / 2.0;

                    let condition = if config.0 == "evaluations" {
                        //TODO: change after implementing proper condition
                        conditions::EveryN::new(config.1 as u32, ValueOf::<Evaluations>::new())
                    } else {
                        conditions::LessThanN::new(config.1, NormalizedDiversityLens::<MinimumIndividualDistance>::new())
                    };

                    let replacement = if config.2 == "best" {
                        replacement::pso::ReplaceNBestPSO::new(new_pop, v_max)
                    } else if config.2 == "worst" {
                        replacement::pso::ReplaceNWorstPSO::new(new_pop, v_max)
                    } else {
                        replacement::pso::ReplaceNRandomPSO::new(new_pop, v_max)
                    };

                    // This is the main setup of the algorithm
                    let conf: Configuration<Instance> = mine_explosion_pso(
                        evaluations,
                        pop_size,
                        inertia_weight, // Weight
                        c1, // C1
                        c2, // C2
                        v_max,
                        condition, // exploration mechanism condition
                        new_pop, // number of new solutions the exploration mechanism generates
                        config.3.parse().unwrap(), // center solution that provides basis for generating new solutions
                        replacement, // replacement operator applied after exploration mechanism
                    );

                    let output = format!("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                                         run,
                                         "_",
                                         instance.name(),
                                         "_",
                                         pop_size,
                                         "_",
                                         inertia_weight,
                                         "_",
                                         c1,
                                         "_",
                                         c2,
                                         "_",
                                         config.0,
                                         "_",
                                         config.1,
                                         "_",
                                         new_pop,
                                         "_",
                                         config.3,
                                         "_",
                                         config.2,
                    );

                    let data_dir = Arc::new(PathBuf::from(&folder));
                    fs::create_dir_all(data_dir.as_ref()).expect("TODO: panic message");

                    let experiment_desc = output;
                    let log_file = data_dir.join(format!("{}.cbor", experiment_desc));

                    // This executes the algorithm
                    let setup = conf.optimize_with(&instance, |state: &mut State<_>| -> ExecResult<()> {
                        state.insert_evaluator(evaluator);
                        state.insert(Random::new(seed));
                        state.configure_log(|con| {
                            con
                                .with_many(
                                    conditions::EveryN::iterations(1),
                                    [
                                        ValueOf::<common::Evaluations>::entry(),
                                        BestObjectiveValueLens::entry(),
                                        NormalizedDiversityLens::<MinimumIndividualDistance>::entry(),
                                    ],
                                )
                            ;
                            Ok(())
                        })
                    });
                    setup.unwrap().log().to_cbor(log_file).expect("TODO: panic message");
                }
            }

        });
    Ok(())
}