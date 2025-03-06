#[path = "../../algorithms/mod.rs"]
mod algorithms;

use mahf::{prelude::*, configuration::Configuration, Random,
           lens::common::{BestObjectiveValueLens},
};
use mahf_coco::{Instance, AcceleratedEvaluator, Suite, Context, Options, backends::C, Name::Bbob};

use std::{
    fs::{self},
    path::PathBuf,
    sync::{Arc},
};
use std::time::Instant;
use once_cell::sync::Lazy;
use clap::Parser;
use crate::algorithms::shade::shade;

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
    
    /// Number of differences, 1 or 2
    #[arg(long, default_value_t = 1)]
    y: u32,
    
    /// Crossover operator, bin or exp
    #[arg(long, default_value = "bin")]
    crossover: String,
    
    /// History size, 1 to maximum number of iterations; tuning in [1, 1000]
    #[arg(long, default_value_t = 100)]
    history: usize,
}


fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let seed = args.seed;
    let _inst = args.inst;
    let pop_size = args.population_size;
    let functions = args.function;
    let instances = args.instance;
    let dimensions: usize = args.dimensions;
    let y = args.y;
    let cr_operator = args.crossover;
    let history = args.history;

    let folder = format!("data/irace_shade/d{:?}_p{:?}", dimensions, pop_size);

    // define remaining algorithmic parameters according to Tanabe and Fukunaga 2013
    let max_archive = pop_size as usize;
    let p_min = 2.0 / pop_size as f64;
    let f = 0.5;
    let cr = 0.5;
    
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
        
        let crossover = if cr_operator == "exp" {
            recombination::de::DEExponentialCrossover::new(cr).unwrap()
        } else {
            recombination::de::DEBinomialCrossover::new(cr).unwrap()
        };
        
        // This is the main setup of the algorithm
        let conf: Configuration<Instance> = shade(
            evaluations,
            pop_size,
            y, // number of individuals to select for mutation
            p_min, // minimum for parameter selecting the pbest
            max_archive, // maximum size of archive
            history, // maximum length of history for F and CR adaptation
            f, // initial value of F; of no consequence when using SHADEAdaptation
            crossover, // exp or bin
        );
        
        let output = format!("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                             instance.name(),
                             "_",
                             pop_size,
                             "_",
                             y,
                             "_",
                             p_min,
                             "_",
                             max_archive,
                             "_",
                             history,
                             "_",
                             f,
                             "_",
                             cr_operator
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