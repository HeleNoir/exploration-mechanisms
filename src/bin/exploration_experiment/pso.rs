#[path = "../../algorithms/mod.rs"]
mod algorithms;

use mahf::{
    configuration::Configuration, lens::common::BestObjectiveValueLens, prelude::*, Random,
};
use mahf_coco::{backends::C, AcceleratedEvaluator, Context, Instance, Name::Bbob, Options, Suite};

use crate::algorithms::pso::basic_pso;
use clap::Parser;
use mahf::components::measures::diversity::{MinimumIndividualDistance, NormalizedDiversityLens};
use mahf::problems::LimitedVectorProblem;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::time::Instant;
use std::{
    fs::{self},
    path::PathBuf,
    sync::Arc,
};

static CONTEXT: Lazy<Context<C>> = Lazy::new(Context::default);

#[derive(Parser)]
#[clap(version, about)]
struct Args {
    /// Number of BBOB function
    #[arg(long, default_value_t = 1)]
    function: usize,

    /// Dimensions of BBOB function
    #[arg(long, default_value_t = 10)]
    dimensions: usize,

    /// Population size of algorithm
    #[arg(long, default_value_t = 50)]
    population_size: u32,

    /// Inertia weight of PSO; 0.0 to 1.0
    #[arg(long, default_value_t = 0.9)]
    inertia_weight: f64,

    /// C1 of PSO; 0.0 to 2.5
    #[arg(long, default_value_t = 0.5)]
    c1: f64,

    /// C2 of PSO; 0.0 to 2.5
    #[arg(long, default_value_t = 0.5)]
    c2: f64,
}
//TODO Add tuning results as default parameter

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    let pop_size = args.population_size;
    let functions = args.function;
    let dimensions: usize = args.dimensions;
    let inertia_weight: f64 = args.inertia_weight;
    let c1: f64 = args.c1;
    let c2: f64 = args.c2;

    // Start timing execution
    let start = Instant::now();

    // set number of evaluations
    let evaluations: u32 = (10000 * dimensions) as u32;

    let folder = format!("data/PSO/d{:?}", dimensions);

    // set number of runs per instance
    let runs: [usize; 25] = (1..=25)
        .collect::<Vec<_>>()
        .try_into()
        .expect("wrong size iterator");
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
        .zip(
            std::iter::repeat(evaluators)
                .take(runs.len())
                .collect::<Vec<_>>(),
        )
        .for_each(|(run, evaluator)| {
            
            for (i, (instance, eval)) in problems.iter().zip(evaluator.iter()).enumerate() {
                let evaluator = eval.clone();

                let seed = seeds[run - 1][i];

                let bounds = instance.domain();
                let lower = bounds[0].start.clone();
                let upper = bounds[0].end.clone();
                let v_max = (upper - lower) / 2.0;

                // This is the main setup of the algorithm
                let conf: Configuration<Instance> = basic_pso(
                    evaluations,
                    pop_size,
                    inertia_weight, // Weight
                    c1,             // C1
                    c2,             // C2
                    v_max,
                );

                let output = format!(
                    "{}{}{}{}{}{}{}{}{}{}{}",
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
                );

                let data_dir = Arc::new(PathBuf::from(&folder));
                fs::create_dir_all(data_dir.as_ref()).expect("TODO: panic message");

                let experiment_desc = output;
                let log_file = data_dir.join(format!("{}.cbor", experiment_desc));

                // This executes the algorithm
                let setup =
                    conf.optimize_with(&instance, |state: &mut State<_>| -> ExecResult<()> {
                        state.insert_evaluator(evaluator);
                        state.insert(Random::new(seed));
                        state.configure_log(|con| {
                            con.with_many(
                                conditions::EveryN::iterations(1),
                                [
                                    ValueOf::<common::Evaluations>::entry(),
                                    BestObjectiveValueLens::entry(),
                                    NormalizedDiversityLens::<MinimumIndividualDistance>::entry(),
                                ],
                            );
                            Ok(())
                        })
                    });
                
                let results = setup.unwrap();
                results
                    .log()
                    .to_cbor(log_file)
                    .expect("TODO: panic message");
                
                // Measure elapsed time
                let duration = start.elapsed();

                println!(
                    "\n{:?}\n{}",
                    results.best_objective_value().unwrap(),
                    duration.as_secs_f64()
                );
            }
        });
    Ok(())
}
