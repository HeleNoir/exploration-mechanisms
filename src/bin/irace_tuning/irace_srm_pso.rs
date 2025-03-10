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
use mahf::components::measures::diversity::{MinimumIndividualDistance, NormalizedDiversityLens};
use mahf::conditions::common::PartialEqChecker;
use mahf::prelude::common::Evaluations;
use mahf::problems::LimitedVectorProblem;
use crate::algorithms::pso_srm::srm_pso;

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

    /// Exploration condition; evaluations or diversity
    #[arg(long, default_value = "evaluations")]
    exploration: String,

    /// Exploration parameter
    #[arg(long, default_value_t = 0.05)]
    exp_param: f64,

    /// Population size of exploration mechanism, number of individuals that will be replaced; 1 to pop_size
    #[arg(long, default_value_t = 5)]
    new_pop: u32,

    /// Solutions to be replaced; best, worst or random
    #[arg(long, default_value = "best")]
    replacement: String,
    
    /// Solution to be used as center; best, random_new or random_solution
    #[arg(long, default_value = "best")]
    center: String,
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
    let exploration = args.exploration;
    let exp_param: f64 = args.exp_param;
    let new_pop: u32 = args.new_pop;
    let replacement = args.replacement;
    let center = args.center;
    
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
        
        let center_solution = center.parse().unwrap();

        let condition = if exploration == "evaluations" {
            let eval_interval = exp_param * evaluations as f64;
            conditions::StagnationForN::new(eval_interval as usize, ValueOf::<Evaluations>::new(), BestObjectiveValueLens::new(), PartialEqChecker::new())
        } else {
            conditions::LessThanN::new(exp_param, NormalizedDiversityLens::<MinimumIndividualDistance>::new())
        };

        let replacement_operator = if replacement == "best" {
            replacement::pso::ReplaceNBestPSO::new(new_pop, v_max)
        } else if replacement == "worst" {
            replacement::pso::ReplaceNWorstPSO::new(new_pop, v_max)
        } else {
            replacement::pso::ReplaceNRandomPSO::new(new_pop, v_max)
        };

        // This is the main setup of the algorithm
        let conf: Configuration<Instance> = srm_pso(
            evaluations,
            pop_size,
            inertia_weight, // Weight
            c1, // C1
            c2, // C2
            v_max,
            condition, // exploration mechanism condition
            new_pop, // number of new solutions the exploration mechanism generates
            center_solution, // center solution that provides basis for generating new solutions
            replacement_operator, // replacement operator applied after exploration mechanism
        );

        // This executes the algorithm
        let setup = conf.optimize_with(&instance, |state: &mut State<_>| -> ExecResult<()> {
            state.insert_evaluator(evaluator);
            state.insert(Random::new(seed));
            Ok(())
        });
        
        let results = setup.unwrap();

        // Measure elapsed time
        let duration = start.elapsed();

        println!("\n{:?}\n{}", results.best_objective_value().unwrap(), duration.as_secs_f64());
    }
    Ok(())
}