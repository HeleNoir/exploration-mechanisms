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
use std::convert::TryInto;
use once_cell::sync::Lazy;
use clap::Parser;
use itertools::iproduct;
use mahf::problems::LimitedVectorProblem;
use rayon::prelude::*;
use crate::algorithms::shade::shade;

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
}


fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let pop_size = args.population_size;
    let functions = args.function;
    let dimensions: usize = args.dimensions;
    
    let folder = format!("data/basic_shade/d{:?}_p{:?}", dimensions, pop_size);

    // set number of runs per instance
    let runs: [usize; 25] = (1..=25).collect::<Vec<_>>()
        .try_into().expect("wrong size iterator");
    // set number of evaluations
    let evaluations: u32 = (10000 * dimensions) as u32;
    
    // define remaining algorithmic parameters
    let y = 2;
    let history = 100;
    let max_archive = pop_size as usize;
    let crossovers = ["exp", "bin"];
    let p_min = 2.0 / pop_size as f64;
    let f = 0.5;
    let cr = 0.5;
    
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
            
            for c in crossovers.iter() {
                for (i, (instance, eval)) in problems.iter().zip(evaluator.iter()).enumerate() {
                    let evaluator = eval.clone();

                    let seed = seeds[run - 1][i];

                    let crossover = if *c == "exp" {
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

                    let output = format!("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                                         run,
                                         "_",
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
                                         c
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