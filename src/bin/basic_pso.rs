pub mod particle_swarm;

use crate::particle_swarm::behaviour_pso;

use mahf::{prelude::*, configuration::Configuration, Random,
           lens::common::{BestObjectiveValueLens, ObjectiveValuesLens, PopulationLens},
           components::measures::{diversity::{NormalizedDiversityLens, DimensionWiseDiversity, PairwiseDistanceDiversity, MinimumIndividualDistance, RadiusDiversity},
                                  improvement::{FitnessImprovement, TotalImprovementLens},
                                  stepsize::{EuclideanStepSize, IndividualStepSizeLens, MeanStepSizeLens, StepSizeVarianceLens}},
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
use mahf::lens::common::BestSolutionLens;
use mahf::problems::LimitedVectorProblem;
use rayon::prelude::*;

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

    let folder = format!("data/basic_PSO/d{:?}_p{:?}", dimensions, pop_size);

    // set number of runs per instance
    let runs = [1, 2, 3, 4, 5];
    // set number of evaluations and iterations
    let evaluations: u32 = (10000 * dimensions) as u32;
    let iterations = (evaluations - pop_size)/pop_size;

    // set algorithm parameters
    let weights = [0.1, 0.3, 0.5, 0.7, 1.0];
    let c1s = [0.5, 1.0, 1.7];
    let c2s = [0.5, 1.0, 1.7];

    let configs: Vec<_> = iproduct!(weights, c1s, c2s).collect();

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

                    let conf: Configuration<Instance> = behaviour_pso(
                        evaluations,
                        pop_size,
                        config.0, // Weight
                        config.1, // C1
                        config.2, // C2
                        v_max,
                    );

                    let output = format!("{}{}{}{}{}{}{}{}{}{}{}",
                                         run,
                                         "_",
                                         instance.name(),
                                         "_",
                                         pop_size,
                                         "_",
                                         config.0,
                                         "_",
                                         config.1,
                                         "_",
                                         config.2,
                    );

                    let data_dir = Arc::new(PathBuf::from(&folder));
                    fs::create_dir_all(data_dir.as_ref()).expect("TODO: panic message");

                    let experiment_desc = output;
                    let log_file = data_dir.join(format!("{}.cbor", experiment_desc));

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
                                        ObjectiveValuesLens::entry(),
                                        NormalizedDiversityLens::<DimensionWiseDiversity>::entry(),
                                        NormalizedDiversityLens::<PairwiseDistanceDiversity>::entry(),
                                        NormalizedDiversityLens::<MinimumIndividualDistance>::entry(),
                                        NormalizedDiversityLens::<RadiusDiversity>::entry(),
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