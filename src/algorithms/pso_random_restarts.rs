use mahf::{prelude::*,
           components::{initialization,
                        measures::{diversity::{DimensionWiseDiversity, PairwiseDistanceDiversity, MinimumIndividualDistance, RadiusDiversity},}},
           configuration::Configuration, logging::Logger,
           problems::{LimitedVectorProblem, SingleObjectiveProblem, KnownOptimumProblem}};


pub fn random_restart_pso<P>(
    evaluations: u32,
    population_size: u32,
    restart_interval: u32,
    w: f64,
    c1: f64,
    c2: f64,
    v_max: f64,
) -> Configuration<P>
where P: SingleObjectiveProblem + LimitedVectorProblem<Element = f64> + KnownOptimumProblem,
{
    Configuration::builder()
        .do_(initialization::RandomSpread::new(population_size))
        .evaluate()
        .update_best_individual()
        .do_(Box::from(swarm::pso::ParticleSwarmInit::new(v_max)))
        .do_(DimensionWiseDiversity::new())
        .do_(PairwiseDistanceDiversity::new())
        .do_(MinimumIndividualDistance::new())
        .do_(RadiusDiversity::new())
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
                    .do_(boundary::Saturation::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(DimensionWiseDiversity::new())
                    .do_(PairwiseDistanceDiversity::new())
                    .do_(MinimumIndividualDistance::new())
                    .do_(RadiusDiversity::new())
                    .do_(swarm::pso::ParticleSwarmUpdate::new())
                    .do_(Logger::new())
                    .if_(conditions::EveryN::iterations(restart_interval), |builder| {
                        builder
                            .do_(initialization::RandomSpread::new(population_size))
                            .evaluate()
                            .update_best_individual()
                            .do_(Box::from(swarm::pso::ParticleSwarmInit::new(v_max)))
                            .do_(DimensionWiseDiversity::new())
                            .do_(PairwiseDistanceDiversity::new())
                            .do_(MinimumIndividualDistance::new())
                            .do_(RadiusDiversity::new())
                            .do_(Logger::new())
                    })
            },
        )
        .build()
}