use mahf::{prelude::*,
           components::{initialization,
                        measures::{diversity::{MinimumIndividualDistance}, }},
           configuration::Configuration, logging::Logger,
           problems::{LimitedVectorProblem, SingleObjectiveProblem, KnownOptimumProblem}};
use mahf::identifier::Global;

pub fn stepped_leader_pso<P>(
    evaluations: u32,
    population_size: u32,
    w: f64,
    c1: f64,
    c2: f64,
    v_max: f64,
    condition: Box<dyn Condition<P>>,
    new_pop: u32,
    leader: String,
    replacement: Box<dyn Component<P>>,
) -> Configuration<P>
where P: SingleObjectiveProblem + LimitedVectorProblem<Element = f64> + KnownOptimumProblem,
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
                    .if_else_(condition, |builder| {
                        builder
                            .do_(selection::All::new())
                            .do_(swarm::lsa::NegativelyChargedSteppedLeader::new(new_pop, leader))
                            .do_(boundary::CompleteOneTailedNormalCorrection::new())
                            .do_(replacement)
                    }, |builder| {
                        builder
                            .do_(Box::from(swarm::pso::ParticleVelocitiesUpdate::new(
                                w,
                                c1,
                                c2,
                                v_max,
                            )))
                            .do_(boundary::CompleteOneTailedNormalCorrection::new())
                    })
                    .evaluate_with::<Global>()
                    .update_best_individual()
                    .do_(MinimumIndividualDistance::new())
                    .do_(swarm::pso::ParticleSwarmUpdate::new())
                    .do_(Logger::new())
            },
        )
        .build()
}