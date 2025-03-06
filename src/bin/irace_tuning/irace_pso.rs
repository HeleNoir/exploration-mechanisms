#[path = "../../algorithms/mod.rs"]
mod algorithms;

use mahf::{prelude::*, configuration::Configuration, Random,
           lens::common::{BestObjectiveValueLens}};
use mahf_coco::{Instance, AcceleratedEvaluator, Suite, Context, Options, backends::C, Name::Bbob};

use std::{
    fs::{self},
    path::PathBuf,
    sync::{Arc},
};
use std::time::Instant;
use once_cell::sync::Lazy;
use clap::Parser;
use mahf::problems::LimitedVectorProblem;
use crate::algorithms::pso::basic_pso;

static CONTEXT: Lazy<Context<C>> = Lazy::new(Context::default);

#[derive(Parser)]
#[clap(version, about)]
struct Args {
    /// Seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    
    /// Instance for irace
    #[arg(long, default_value = "1")]
    inst: String,
    
    /// Number of BBOB function
    #[arg(long, default_value_t = 1)]
    function: usize,

    /// Instance of BBOB function
    #[arg(long, default_value_t = 6)]
    instance: usize,

    /// Dimensions of BBOB function
    #[arg(long, default_value_t = 10)]
    dimensions: usize,

    /// Population size of algorithm
    #[arg(long, default_value_t = 50)]
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
}


fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let seed = args.seed;
    let _inst = args.inst;
    let pop_size = args.population_size;
    let functions = args.function;
    let instances = args.instance;
    let dimensions: usize = args.dimensions;
    let inertia_weight: f64 = args.inertia_weight;
    let c1: f64 = args.c1;
    let c2: f64 = args.c2;

    let folder = format!("data/irace_PSO/d{:?}_p{:?}", dimensions, pop_size);

    // Start timing execution
    let start = Instant::now();
    
    // set number of evaluations
    let evaluations: u32 = (10000 * dimensions) as u32;
    
    let options = Options::new()
        .with_dimensions([dimensions])
        .with_function_indices([functions])
        .with_instance_indices([instances]);
    let mut suite = Suite::with_options(Bbob, None, Some(&options)).unwrap();

    while let Some(instance) = suite.next() {
        let evaluator = AcceleratedEvaluator::new(&CONTEXT, &mut suite, &instance);

        let bounds = instance.domain();
        let lower = bounds[0].start.clone();
        let upper = bounds[0].end.clone();
        let v_max = (upper - lower) / 2.0;

        // This is the main setup of the algorithm
        let conf: Configuration<Instance> = basic_pso(
            evaluations,
            pop_size,
            inertia_weight, // Weight
            c1, // C1
            c2, // C2
            v_max,
        );
        
        let output = format!("{}{}{}{}{}{}{}{}{}",
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
        let setup = conf.optimize_with(&instance, |state: &mut State<_>| -> ExecResult<()> {
            state.insert_evaluator(evaluator);
            state.insert(Random::new(seed));
            state.configure_log(|con| {
                con
                    .with_many(
                        conditions::EveryN::iterations(100),
                        [
                            ValueOf::<common::Evaluations>::entry(),
                            BestObjectiveValueLens::entry(),
                        ],
                    )
                ;
                Ok(())
            })
        });
        let results = setup.unwrap();
        results.log().to_cbor(log_file).expect("TODO: panic message");

        // Measure elapsed time
        let duration = start.elapsed();
        
        println!("\n{:?}\n{}", results.best_objective_value().unwrap(), duration.as_secs_f64());
    }
    Ok(())
}