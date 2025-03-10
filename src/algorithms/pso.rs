use mahf::{prelude::*,
           components::{initialization,
                        measures::{diversity::{MinimumIndividualDistance}, }},
           configuration::Configuration, logging::Logger,
           problems::{LimitedVectorProblem, SingleObjectiveProblem}};


pub fn basic_pso<P>(
    evaluations: u32,
    population_size: u32,
    w: f64,
    c1: f64,
    c2: f64,
    v_max: f64,
) -> Configuration<P>
where P: SingleObjectiveProblem + LimitedVectorProblem<Element = f64>,
{
    Configuration::builder()
        .do_(initialization::RandomSpread::new(population_size))
        .evaluate()
        .update_best_individual()
        .do_(Box::from(swarm::pso::ParticleSwarmInit::new(v_max)))
        .do_(MinimumIndividualDistance::new())
        .do_(Logger::new())
        .while_(
            conditions::LessThanN::evaluations(evaluations),
            |builder| {
                builder
                    .do_(Box::from(swarm::pso::ParticleVelocitiesUpdate::new(
                        w,
                        c1,
                        c2,
                        v_max,
                    )))
                    .do_(boundary::CosineCorrection::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(MinimumIndividualDistance::new())
                    .do_(swarm::pso::ParticleSwarmUpdate::new())
                    .do_(Logger::new())
            },
        )
        .build()
}