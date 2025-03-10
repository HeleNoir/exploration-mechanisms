use mahf::{prelude::*,
           components::{initialization,
                        measures::{diversity::{MinimumIndividualDistance}, }},
           configuration::Configuration, logging::Logger,
           problems::{LimitedVectorProblem, SingleObjectiveProblem}};
use eyre::WrapErr;

pub fn shade<P>(
    evaluations: u32,
    population_size: u32,
    y: u32,
    p_min: f64,
    max_archive: usize,
    history: usize,
    f: f64,
    crossover: Box<dyn Component<P>>,
) -> Configuration<P>
where P: SingleObjectiveProblem + LimitedVectorProblem<Element = f64>,
{
    Configuration::builder()
        .do_(initialization::RandomSpread::new(population_size))
        .evaluate()
        .update_best_individual()
        .do_(MinimumIndividualDistance::new())
        .do_(mapping::de::SHADEAdaptationInit::new(history).expect("failed to initialise SHADE adaptation states"))
        .do_(Logger::new())
        .while_(
            conditions::LessThanN::evaluations(evaluations),
            |builder|{
                builder
                    .do_(Box::from(mapping::de::SHADEAdaptation::new().expect("failed to construct SHADE Adaptation")))
                    .do_(selection::de::SHADECurrentToPBest::new(y, p_min, population_size as usize, max_archive).wrap_err("failed to construct DE selection").unwrap())
                    .do_(mutation::de::DEMutation::new(y, f).wrap_err("failed to construct DE mutation").unwrap())
                    .do_(crossover)
                    .do_(boundary::CosineCorrection::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(components::archive::DEKeepParentsArchiveUpdate::new(max_archive))
                    .do_(mapping::de::SHADEAdaptationHistoryUpdate::new().expect("failed to construct SHADE AdaptationHistory"))
                    .do_(replacement::KeepBetterAtIndex::new())
                    .do_(MinimumIndividualDistance::new())
                    .do_(Logger::new())
            }
        )
        .build()
}