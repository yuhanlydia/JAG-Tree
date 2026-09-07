"""Finite PBPF mechanism tests."""

from __future__ import annotations

import unittest

import numpy as np


class PBPFPhaseZeroTests(unittest.TestCase):
    def test_canonical_programs_have_64_distinct_truth_tables(self) -> None:
        from coding_opsd.pbpf.dsl import canonical_programs

        programs = canonical_programs()
        self.assertEqual(len(programs), 64)
        self.assertEqual(len({program.outputs for program in programs}), 64)

    def test_mutation_inverse_round_trip_and_pinned_prior(self) -> None:
        from coding_opsd.pbpf.dsl import BUG_PRIOR, Bug, canonical_programs, inverse_mutations, mutate

        program = canonical_programs()[3]
        for bug in Bug:
            for site in bug.sites:
                mutated = mutate(program, bug, site)
                self.assertIn((bug, site), inverse_mutations(program, mutated))
        np.testing.assert_allclose(BUG_PRIOR, [.20, .12, .12, .12, .11, .11, .11, .11])
        with self.assertRaises(TypeError):
            BUG_PRIOR[0] = .0

    def test_candidate_likelihood_marginalizes_same_family_source_ambiguity(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, canonical_programs, mutate
        from coding_opsd.pbpf.posterior import candidate_likelihood

        programs = canonical_programs()
        candidate = mutate(programs[0], Bug.CORRECT, 0)
        direct = candidate_likelihood(0, candidate)
        family_average = np.mean(
            [candidate_likelihood(source_h, candidate) for source_h in range(8)]
        )
        contaminated = candidate_likelihood(
            0, candidate, candidate_source_contamination=.8
        )
        self.assertAlmostEqual(contaminated, .2 * direct + .8 * family_average)
        self.assertGreater(
            candidate_likelihood(7, candidate, candidate_source_contamination=.8),
            0.0,
        )

    def test_test_order_is_seeded_independently(self) -> None:
        from coding_opsd.pbpf.dsl import deterministic_test_order

        inputs = tuple(range(16))
        self.assertEqual(deterministic_test_order("one", inputs, 7), deterministic_test_order("one", inputs, 7))
        self.assertNotEqual(deterministic_test_order("one", inputs, 7), deterministic_test_order("two", inputs, 7))

    def test_numeric_two_is_a_passing_value_not_an_exception(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Outcome, canonical_programs, mutate, outcome_for

        program = canonical_programs()[2]  # h(0) == the ordinary integer 2
        self.assertEqual(outcome_for(program, program, 0), Outcome.PASS)
        exceptional = mutate(program, Bug.EXCEPTION, 0)
        self.assertIsInstance(exceptional(0), Outcome)
        self.assertNotEqual(exceptional, program)
        self.assertEqual(outcome_for(program, exceptional, 0), Outcome.EXCEPTION)

    def test_program_rejects_pass_wrong_sentinels_but_accepts_numeric_one_two(self) -> None:
        from coding_opsd.pbpf.dsl import Outcome, Program

        self.assertEqual(Program((1, 2) + (0,) * 62).outputs[:2], (1, 2))
        with self.assertRaises(ValueError):
            Program((Outcome.PASS,) + (0,) * 63)
        with self.assertRaises(ValueError):
            Program((Outcome.WRONG,) + (0,) * 63)
        with self.assertRaises(ValueError):
            Program((1.9,) + (0,) * 63)
        with self.assertRaises(ValueError):
            Program((True,) + (0,) * 63)

    def test_prefix_has_no_grader_fields(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate

        h = canonical_programs()[0]
        episode = Episode.from_candidates("leak", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=4)
        prefix = episode.prefix_view(1)
        self.assertEqual(prefix.observed_outcomes.shape, (1, 1))
        self.assertEqual(prefix.observed_count, 1)
        self.assertEqual(prefix.test_order, episode.test_order)
        self.assertFalse(hasattr(prefix, "true_h"))
        self.assertFalse(hasattr(prefix, "outcome_matrix"))

    def test_prefix_view_rejects_supplied_future_outcomes(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate

        h = canonical_programs()[0]
        episode = Episode.from_candidates("future", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=4)
        with self.assertRaises(ValueError):
            episode.prefix_view(0, outcomes=episode.outcome_matrix)
        with self.assertRaises(ValueError):
            episode.prefix_view(1.9)
        with self.assertRaises(ValueError):
            episode.prefix_view(1, outcomes=np.asarray([[1.9]]))
        with self.assertRaises(ValueError):
            episode.prefix_view(True)

    def test_exact_normalized_sequential_and_bruteforce_match(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.posterior import brute_force_h_marginal, exact_posterior, sequential_update

        h = canonical_programs()[9]
        episode = Episode.from_candidates("posterior", 9, (mutate(h, Bug.PLUS_ONE, 0), mutate(h, Bug.BOUNDARY_FLIP, 3)), order_seed=8)
        prefix = episode.prefix_view(4)
        batch = exact_posterior(episode, prefix)
        np.testing.assert_allclose(batch.sum(), 1.0, atol=1e-8)
        np.testing.assert_allclose(sequential_update(episode, prefix), batch, atol=1e-8)
        np.testing.assert_allclose(brute_force_h_marginal(episode, prefix), batch, atol=1e-8)

    def test_sequential_rejects_foreign_or_reordered_prefix(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, PrefixView, canonical_programs, mutate
        from coding_opsd.pbpf.posterior import sequential_update

        h = canonical_programs()[9]
        episode = Episode.from_candidates("prefix", 9, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=8)
        prefix = episode.prefix_view(2)
        foreign = PrefixView("other", prefix.candidate_programs, prefix.inputs, prefix.test_order, prefix.observed_outcomes)
        with self.assertRaises(ValueError):
            sequential_update(episode, foreign)
        reordered = PrefixView(prefix.episode_id, prefix.candidate_programs, prefix.inputs, prefix.test_order[::-1], prefix.observed_outcomes)
        with self.assertRaises(ValueError):
            sequential_update(episode, reordered)

    def test_predictive_is_explicit_h_marginal_and_candidate_equivariant(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.posterior import exact_posterior, predictive

        h = canonical_programs()[11]
        candidates = (mutate(h, Bug.PLUS_ONE, 0), mutate(h, Bug.EXCEPTION, 2))
        episode = Episode.from_candidates("predict", 11, candidates, order_seed=3)
        probability = exact_posterior(episode, episode.prefix_view(1))
        predicted = predictive(probability, episode, [1, 2])
        explicit = np.zeros_like(predicted)
        for h_id, weight in enumerate(probability):
            for g, candidate in enumerate(candidates):
                for index, position in enumerate([1, 2]):
                    explicit[g, index, episode.outcome_at(h_id, candidate, position)] += weight
        explicit = np.maximum(explicit, 1e-8)
        explicit /= explicit.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(predicted, explicit, atol=1e-8)
        permuted = Episode.from_candidates("predict", 11, candidates[::-1], order_seed=3)
        np.testing.assert_allclose(predicted[::-1], predictive(probability, permuted, [1, 2]), atol=1e-8)
        with self.assertRaises(ValueError):
            predictive(probability, episode, [1], floor=float("nan"))
        with self.assertRaises(ValueError):
            predictive(probability, episode, [1], floor=float("inf"))

    def test_impossible_evidence_and_randomized_hpd(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, Outcome, canonical_programs, mutate
        from coding_opsd.pbpf.posterior import ImpossibleEvidenceError, exact_map, exact_posterior, randomized_hpd

        h = canonical_programs()[0]
        episode = Episode.from_candidates("bad", 0, (mutate(h, Bug.CORRECT, 0),), order_seed=1)
        prefix = episode.prefix_view(1, outcomes=np.asarray([[Outcome.EXCEPTION]], dtype=int))
        with self.assertRaises(ImpossibleEvidenceError):
            exact_posterior(episode, prefix)
        prob = np.asarray([.5, .3, .2])
        inclusion = randomized_hpd(prob, .9)
        np.testing.assert_allclose(np.dot(prob, inclusion), .9, atol=1e-12)
        np.testing.assert_allclose(inclusion, [1.0, 1.0, .5], atol=1e-12)
        with self.assertRaises(ValueError):
            exact_map(np.ones(64))

    def test_particles_have_finite_normalized_weights_and_resampling_is_deterministic(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter, systematic_resample

        h = canonical_programs()[7]
        episode = Episode.from_candidates("particles", 7, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=2)
        particle_filter = ParticleFilter.initialize(episode, particles=8, seed=12)
        particle_filter.observe(0, episode.prefix_view(1).observed_outcomes[:, 0])
        self.assertTrue(np.isfinite(particle_filter.log_weights).all())
        np.testing.assert_allclose(particle_filter.weights.sum(), 1.0)
        first = systematic_resample(np.asarray([.1, .2, .7]), seed=5)
        self.assertEqual(first.tolist(), systematic_resample(np.asarray([.1, .2, .7]), seed=5).tolist())
        self.assertEqual(len(first), 3)
        self.assertEqual(np.bincount(first, minlength=3).tolist(), [0, 1, 2])

    def test_particle_initial_weights_match_candidate_conditioned_proposal(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter
        from coding_opsd.pbpf.posterior import candidate_likelihood

        h = canonical_programs()[0]
        episode = Episode.from_candidates("proposal", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=5)
        particle_filter = ParticleFilter.initialize(episode, particles=8, seed=3)
        compatible = [h_id for h_id in range(64) if candidate_likelihood(h_id, episode.candidate_programs[0]) > 0]
        raw = np.asarray([candidate_likelihood(state.h, episode.candidate_programs[0]) / 64.0 / (1.0 / len(compatible)) for state in particle_filter.states])
        np.testing.assert_allclose(particle_filter.weights, raw / raw.sum())

    def test_stratified_particle_initialization_covers_small_candidate_support(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter

        programs = canonical_programs()
        episode = Episode.from_candidates(
            "stratified",
            0,
            (mutate(programs[0], Bug.CORRECT, 0),),
            order_seed=5,
            candidate_source_contamination=.8,
        )
        particle_filter = ParticleFilter.initialize(episode, particles=16, seed=3)
        self.assertEqual({state.h for state in particle_filter.states}, set(range(8)))

    def test_particle_mh_rejuvenation_redraws_compatible_explanations(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter
        from coding_opsd.pbpf.posterior import compatible_explanations

        h = canonical_programs()[0]
        episode = Episode.from_candidates("mh", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=5)
        particle_filter = ParticleFilter.initialize(episode, particles=4, seed=4)
        particle_filter._resample_and_rejuvenate()
        self.assertEqual(particle_filter.resampling_count, 1)
        for state in particle_filter.states:
            compatible = {(int(bug), site) for bug, site, _ in compatible_explanations(state.h, episode.candidate_programs[0])}
            self.assertIn((state.bugs[0], state.sites[0]), compatible)

    def test_particle_records_pre_resampling_diagnostics(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter, ParticleState

        h = canonical_programs()[0]
        episode = Episode.from_candidates("diagnostics", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=5)
        particle_filter = ParticleFilter.initialize(episode, particles=4, seed=4)
        particle_filter.states = [ParticleState(0, (1,), (0,)) for _ in range(4)]
        particle_filter.log_weights = np.asarray([0.0, -100.0, -100.0, -100.0])
        particle_filter._normalize_log_weights()
        particle_filter.observe(0, episode.outcome_matrix[:, 0])
        diagnostic = particle_filter.diagnostic_history[-1]
        self.assertTrue(diagnostic.resampled)
        self.assertLess(diagnostic.pre_resampling_ess, 2.0)
        self.assertEqual(diagnostic.unique_ancestor_ratio, .25)
        self.assertLessEqual(diagnostic.post_rejuvenation_unique_state_ratio, 1.0)
        self.assertEqual(particle_filter.ess_history[-1], diagnostic.pre_resampling_ess)

    def test_particles_per_correct_equivalence_class_counts_true_h_particles(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter, ParticleState

        h = canonical_programs()[0]
        episode = Episode.from_candidates("true-count", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=5)
        particle_filter = ParticleFilter.initialize(episode, particles=3, seed=4)
        alternative = ParticleState(1, (0,), (0,))
        true = ParticleState(0, (1,), (0,))
        particle_filter.states = [alternative, alternative, alternative]
        self.assertEqual(particle_filter.particles_per_correct_equivalence_class, 0)
        particle_filter.states = [true, alternative, alternative]
        self.assertEqual(particle_filter.particles_per_correct_equivalence_class, 1)
        particle_filter.states = [true, true, alternative]
        self.assertEqual(particle_filter.particles_per_correct_equivalence_class, 2)

    def test_particle_observe_rejects_nonintegral_or_bool_position(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter

        h = canonical_programs()[0]
        episode = Episode.from_candidates("position", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=5)
        for invalid in (0.0, 1.9, False):
            particle_filter = ParticleFilter.initialize(episode, particles=2, seed=4)
            with self.assertRaises(ValueError):
                particle_filter.observe(invalid, episode.outcome_matrix[:, 0])

    def test_collapse_event_payload_retains_prior_resampling_events(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.experiment import _resampling_events
        from coding_opsd.pbpf.particles import ParticleDiagnostic, ParticleFilter

        h = canonical_programs()[0]
        episode = Episode.from_candidates("collapse-events", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=5)
        particle_filter = ParticleFilter.initialize(episode, particles=2, seed=4)
        particle_filter.diagnostic_history.append(ParticleDiagnostic(0, 1.0, True, .5, .5))
        particle_filter.diagnostic_history.append(ParticleDiagnostic(1, 0.0, False, None, None))
        self.assertEqual(_resampling_events(particle_filter), [{"position": 0, "pre_resampling_ess": 1.0, "unique_ancestor_ratio": .5, "post_rejuvenation_unique_state_ratio": .5}])

    def test_particle_h_marginal_rejects_invalid_floor(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter

        h = canonical_programs()[0]
        episode = Episode.from_candidates("floor", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=5)
        particle_filter = ParticleFilter.initialize(episode, particles=2, seed=4)
        for invalid in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                particle_filter.full_support_h_marginal(invalid)

    def test_particle_prediction_floors_outcomes_not_hypothesis_mass(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter, ParticleState

        h = canonical_programs()[0]
        episode = Episode.from_candidates("empirical", 0, (mutate(h, Bug.CORRECT, 0),), order_seed=2)
        particle_filter = ParticleFilter.initialize(episode, particles=1, seed=1)
        particle_filter.states = [ParticleState(0, (0,), (0,))]
        particle_filter.log_weights = np.asarray([0.0])
        marginal = particle_filter.h_marginal()
        self.assertEqual(np.count_nonzero(marginal), 1)
        forecast = particle_filter.predict([0])
        self.assertLess(forecast[0, 0, 1], 1.1e-8)

    def test_particle_collapse_fails_closed(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleCollapseError, ParticleFilter

        h = canonical_programs()[0]
        episode = Episode.from_candidates("collapse", 0, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=1)
        for seed in range(64):
            particle_filter = ParticleFilter.initialize(episode, particles=1, seed=seed)
            predicted = episode.outcome_at(particle_filter.states[0].h, episode.candidate_programs[0], 0)
            if predicted != episode.outcome_matrix[0, 0]:
                with self.assertRaises(ParticleCollapseError):
                    particle_filter.observe(0, episode.outcome_matrix[:, 0])
                break
        else:
            self.fail("fixture did not sample a contradictory particle")

    def test_full_finite_support_is_not_destroyed_by_resampling(self) -> None:
        from coding_opsd.pbpf.experiment import _episode
        from coding_opsd.pbpf.particles import ParticleFilter
        from coding_opsd.pbpf.posterior import exact_posterior

        episode = _episode(303, 66, 4, 61030, .8)
        particle_filter = ParticleFilter.initialize(
            episode, particles=16, seed=303 + 66 * 1009 + 4
        )
        for position in range(4):
            particle_filter.observe(position, episode.outcome_matrix[:, position])

        np.testing.assert_allclose(
            particle_filter.h_marginal(),
            exact_posterior(episode, episode.prefix_view(4)),
            atol=1e-12,
        )

    def test_particles_reject_out_of_order_manifest_observation(self) -> None:
        from coding_opsd.pbpf.dsl import Bug, Episode, canonical_programs, mutate
        from coding_opsd.pbpf.particles import ParticleFilter

        h = canonical_programs()[5]
        episode = Episode.from_candidates("ordered", 5, (mutate(h, Bug.PLUS_ONE, 0),), order_seed=3)
        particle_filter = ParticleFilter.initialize(episode, particles=4, seed=2)
        with self.assertRaises(ValueError):
            particle_filter.observe(1, episode.outcome_matrix[:, 1])

    def test_runner_is_repeated_seed_deterministic(self) -> None:
        from coding_opsd.pbpf.experiment import run_pbpf_phase0

        config = {"phase0": {"episodes": {"test": 3}, "candidates": 2, "prefixes": [0, 1], "particle_sweep": [1, 4]}}
        self.assertEqual(run_pbpf_phase0(config, 99), run_pbpf_phase0(config, 99))

    def test_source_ambiguity_exposes_registered_prefix_four_mixture_effect(self) -> None:
        from coding_opsd.pbpf.experiment import _episode, _scores
        from coding_opsd.pbpf.posterior import exact_map, exact_posterior, predictive

        exact_vs_map = []
        exact_vs_prior = []
        for index in range(64):
            episode = _episode(101, index, 4, 61030, .8)
            posterior = exact_posterior(episode, episode.prefix_view(4))
            positions = tuple(range(4, 64))
            targets = episode.outcome_matrix[:, positions]
            exact_nll, _ = _scores(predictive(posterior, episode, positions), targets)
            map_probability = np.zeros(64)
            map_probability[exact_map(posterior)] = 1.0
            map_nll, _ = _scores(
                predictive(map_probability, episode, positions), targets
            )
            prior_nll, _ = _scores(
                predictive(np.full(64, 1 / 64), episode, positions), targets
            )
            exact_vs_map.append(map_nll - exact_nll)
            exact_vs_prior.append(prior_nll - exact_nll)

        self.assertGreater(float(np.mean(exact_vs_map)), .02)
        self.assertGreater(float(np.mean(exact_vs_prior)), .05)

    def test_runner_reports_prior_gate_inputs_and_rejected_arms(self) -> None:
        from coding_opsd.pbpf.experiment import run_pbpf_phase0

        result = run_pbpf_phase0({"phase0": {"episodes": {"test": 1}, "candidates": 1, "prefixes": [0], "particle_sweep": [1], "arms": ["exact_bayes", "pbpf", "map", "deterministic_state"]}}, 8)
        self.assertIn("prior", {row["arm"] for row in result["rows"]})
        self.assertIn("exact_vs_prior_nll_delta", result["gate_inputs"])
        self.assertEqual(result["rejected_arms"], ["deterministic_state"])

    def test_runner_reports_particle_prefix_diagnostics(self) -> None:
        from coding_opsd.pbpf.experiment import run_pbpf_phase0

        result = run_pbpf_phase0({"runtime": {"profile": "smoke"}, "phase0": {"episodes": {"test": 1}, "candidates": 1, "prefixes": [1], "forecast_horizons": [1], "particle_sweep": [4], "arms": ["pbpf"]}}, 9)
        particle_row = next(row for row in result["rows"] if row["arm"] == "particle_4")
        self.assertIn("pre_resampling_ess", particle_row)
        self.assertIn("resampling_rate", particle_row)
        self.assertIn("particles_per_correct_equivalence_class", particle_row)

    def test_runner_indexes_registered_prefix_horizons_and_fails_formal_closed(self) -> None:
        from coding_opsd.pbpf.experiment import run_pbpf_phase0

        config = {"runtime": {"profile": "formal"}, "phase0": {"episodes": {"test": 1}, "candidates": 1, "prefixes": [4], "forecast_horizons": [1, 8, "all_remaining"], "particle_sweep": [16], "arms": ["exact_bayes", "pbpf", "map"]}}
        result = run_pbpf_phase0(config, 8)
        self.assertEqual({row["horizon"] for row in result["rows"]}, {1, 8, "all_remaining"})
        self.assertIn("4", result["gate_inputs"]["by_prefix_horizon"])
        self.assertIn("p16_to_exact_nll_gap", result["gate_inputs"]["prefix_4"])
        self.assertIn(result["status"], {"FAIL", "INCOMPLETE"})

    def test_formal_complete_flag_cannot_manufacture_pass(self) -> None:
        from coding_opsd.pbpf.experiment import run_pbpf_phase0

        config = {"runtime": {"profile": "formal"}, "phase0": {"formal_sufficiency": {"complete": True}, "episodes": {"test": 1}, "candidates": 1, "prefixes": [4], "forecast_horizons": ["all_remaining"], "particle_sweep": [16], "arms": ["exact_bayes", "pbpf", "map"]}}
        self.assertNotEqual(run_pbpf_phase0(config, 8)["status"], "PASS")

    def test_gate_pairing_keeps_particle16_episode_ids_aligned(self) -> None:
        from coding_opsd.pbpf.experiment import _group_gate_inputs

        group = {"exact": {1: 1.0, 2: 10.0}, "prior": {1: 2.0, 2: 11.0}, "map": {1: 3.0, 2: 12.0}, "particle_16": {2: 10.5}, "hpd": {1: .9, 2: .9}, "collapses": 1, "p16_collapse_ids": set(), "expected_episode_ids": {1, 2}}
        inputs = _group_gate_inputs(group, outer_seed=7)
        self.assertEqual(inputs["paired_p16_to_exact_inputs"], [.5])
        self.assertEqual(inputs["p16_to_exact_nll_gap"], .5)
        self.assertEqual(inputs["p16_missing_episode_ids"], [1])
        self.assertFalse(inputs["p16_complete_coverage"])

    def test_bootstrap_streams_are_outer_seed_derived(self) -> None:
        from coding_opsd.pbpf.experiment import _group_gate_inputs

        group = {"exact": {index: float(index) for index in range(8)}, "prior": {index: float(index) + (index % 3) / 10 for index in range(8)}, "map": {index: float(index) + (index % 4) / 10 for index in range(8)}, "particle_16": {index: float(index) + .2 for index in range(8)}, "hpd": {index: .9 for index in range(8)}, "collapses": 0, "p16_collapse_ids": set(), "expected_episode_ids": set(range(8))}
        first = _group_gate_inputs(group, outer_seed=101, label="prefix=4/horizon=all_remaining", bootstrap_resamples=20)
        self.assertEqual(first, _group_gate_inputs(group, outer_seed=101, label="prefix=4/horizon=all_remaining", bootstrap_resamples=20))
        second = _group_gate_inputs(group, outer_seed=202, label="prefix=4/horizon=all_remaining", bootstrap_resamples=20)
        self.assertNotEqual(first["exact_vs_map_paired_ci"], second["exact_vs_map_paired_ci"])

    def test_formal_status_requires_scale_complete_p16_coverage_and_no_collapse(self) -> None:
        from coding_opsd.pbpf.experiment import _formal_status, _group_gate_inputs

        values = {"exact": {0: 1.0}, "prior": {0: 2.0}, "map": {0: 2.0}, "particle_16": {0: 1.1}, "hpd": {0: .9}, "collapses": 0, "p16_collapse_ids": set(), "expected_episode_ids": {0}}
        inputs = _group_gate_inputs(values, outer_seed=1, bootstrap_resamples=1)
        self.assertIsNotNone(inputs["p16_to_exact_one_sided_upper_95"])
        phase = {"episodes": {"test": 1}, "gate": {"exact_vs_prior_future_nll_nats_per_test": .0, "exact_mixture_vs_exact_map_nll_nats_per_candidate_test_min": .0, "exact_mixture_vs_exact_map_paired_ci_lower_min": -1.0, "p16_gap_to_exact_nll_one_sided_ci_upper_max": 1.0, "exact_map_to_exact_mixture_gap_closed_min": 0.0, "credible_set_nominal": .9}}
        self.assertEqual(_formal_status(phase, inputs, []), "INCOMPLETE")
        registered_inputs = {
            **inputs,
            "expected_episode_ids": list(range(5000)),
            "p16_complete_coverage": True,
        }
        self.assertEqual(_formal_status(phase, registered_inputs, []), "PASS")


if __name__ == "__main__":
    unittest.main()
