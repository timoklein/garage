"""Test root level functions in garage."""
# yapf: disable
import csv
import math
import tempfile

import akro
import dowel
from dowel import logger, tabular
import numpy as np
import pytest
import torch

from garage import (_Default, EnvSpec, EpisodeBatch, log_multitask_performance,
                    log_performance, make_optimizer, rollout, StepType)
from garage.envs import GymEnv

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.policies import DummyPolicy

# yapf: enable


class TestRollout:
    def setup_method(self):
        self.env = GymEnv(DummyBoxEnv(obs_dim=(4, 4), action_dim=(2, 2)))
        self.policy = DummyPolicy(self.env.spec)

    def test_max_episode_length(self):
        # pylint: disable=unsubscriptable-object
        path = rollout(self.env, self.policy, max_episode_length=3)
        assert path["observations"].shape[0] == 3
        assert path["actions"].shape[0] == 3
        assert path["rewards"].shape[0] == 3
        assert path["agent_infos"]["dummy"].shape[0] == 3
        assert path["env_infos"]["dummy"].shape[0] == 3

    def test_deterministic_action(self):
        path = rollout(self.env, self.policy, max_episode_length=5, deterministic=True)
        assert (path["actions"] == 0.0).all()


@pytest.mark.serial
def test_log_performance():
    lengths = np.array([10, 5, 1, 1])
    batch = EpisodeBatch(
        EnvSpec(
            akro.Box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
            akro.Box(np.array([-1.0, -1.0]), np.array([0.0, 0.0])),
        ),
        observations=np.ones((sum(lengths), 3), dtype=np.float32),
        last_observations=np.ones((len(lengths), 3), dtype=np.float32),
        actions=np.zeros((sum(lengths), 2), dtype=np.float32),
        rewards=np.array(
            [
                0.34026529,
                0.58263177,
                0.84307509,
                0.97651095,
                0.81723901,
                0.22631398,
                0.03421301,
                0.97515046,
                0.64311832,
                0.65068933,
                0.17657714,
                0.04783857,
                0.73904013,
                0.41364329,
                0.52235551,
                0.24203526,
                0.43328910,
            ]
        ),
        step_types=np.array(
            [StepType.FIRST]
            + [StepType.MID] * (lengths[0] - 2)
            + [StepType.TERMINAL]
            + [StepType.FIRST]
            + [StepType.MID] * (lengths[1] - 2)
            + [StepType.TERMINAL]
            + [StepType.FIRST]
            + [StepType.FIRST],
            dtype=StepType,
        ),
        env_infos={"success": np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=bool)},
        agent_infos={},
        episode_infos={},
        lengths=lengths,
    )

    log_file = tempfile.NamedTemporaryFile()
    csv_output = dowel.CsvOutput(log_file.name)
    logger.add_output(csv_output)
    log_performance(7, batch, 0.8, prefix="test_log_performance")
    logger.log(tabular)
    logger.dump_output_type(dowel.CsvOutput)
    with open(log_file.name, "r") as file:
        rows = list(csv.DictReader(file))
    res = {k: float(r) for (k, r) in rows[0].items()}
    assert res["test_log_performance/Iteration"] == 7
    assert res["test_log_performance/NumEpisodes"] == 4
    assert math.isclose(res["test_log_performance/SuccessRate"], 0.75)
    assert math.isclose(res["test_log_performance/TerminationRate"], 0.5)
    assert math.isclose(res["test_log_performance/AverageDiscountedReturn"], 1.1131040640673113)
    assert math.isclose(res["test_log_performance/AverageReturn"], 2.1659965525)
    assert math.isclose(res["test_log_performance/StdReturn"], 2.354067152038576)


@pytest.mark.serial
def test_log_multitask_performance_task_name():
    lengths = np.array([10, 5, 1, 1])
    batch = EpisodeBatch(
        EnvSpec(
            akro.Box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
            akro.Box(np.array([-1.0, -1.0]), np.array([0.0, 0.0])),
        ),
        observations=np.ones((sum(lengths), 3), dtype=np.float32),
        last_observations=np.ones((len(lengths), 3), dtype=np.float32),
        actions=np.zeros((sum(lengths), 2), dtype=np.float32),
        rewards=np.array(
            [
                0.34026529,
                0.58263177,
                0.84307509,
                0.97651095,
                0.81723901,
                0.22631398,
                0.03421301,
                0.97515046,
                0.64311832,
                0.65068933,
                0.17657714,
                0.04783857,
                0.73904013,
                0.41364329,
                0.52235551,
                0.24203526,
                0.43328910,
            ]
        ),
        step_types=np.array([StepType.MID] * sum(lengths), dtype=StepType),
        env_infos={
            "success": np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=bool),
            "task_name": np.array(["env1"] * 10 + ["env2"] * 5 + ["env1"] + ["env3"]),
        },
        agent_infos={},
        episode_infos={},
        lengths=lengths,
    )

    log_file = tempfile.NamedTemporaryFile()
    csv_output = dowel.CsvOutput(log_file.name)
    logger.add_output(csv_output)
    log_multitask_performance(7, batch, 0.8)
    logger.log(tabular)
    logger.dump_output_type(dowel.CsvOutput)
    with open(log_file.name, "r") as file:
        rows = list(csv.DictReader(file))
    res = {k: float(r) for (k, r) in rows[0].items()}
    assert res["env1/Iteration"] == 7
    assert res["env2/Iteration"] == 7
    assert res["env3/Iteration"] == 7
    assert res["env1/NumEpisodes"] == 2
    assert res["env2/NumEpisodes"] == 1
    assert res["env3/NumEpisodes"] == 1
    assert math.isclose(res["env1/SuccessRate"], 0.5)
    assert math.isclose(res["env2/SuccessRate"], 1.0)
    assert math.isclose(res["env3/SuccessRate"], 1.0)


@pytest.mark.serial
def test_log_multitask_performance_task_id():
    lengths = np.array([10, 5, 1, 1])
    batch = EpisodeBatch(
        EnvSpec(
            akro.Box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
            akro.Box(np.array([-1.0, -1.0]), np.array([0.0, 0.0])),
        ),
        observations=np.ones((sum(lengths), 3), dtype=np.float32),
        last_observations=np.ones((len(lengths), 3), dtype=np.float32),
        actions=np.zeros((sum(lengths), 2), dtype=np.float32),
        rewards=np.array(
            [
                0.34026529,
                0.58263177,
                0.84307509,
                0.97651095,
                0.81723901,
                0.22631398,
                0.03421301,
                0.97515046,
                0.64311832,
                0.65068933,
                0.17657714,
                0.04783857,
                0.73904013,
                0.41364329,
                0.52235551,
                0.24203526,
                0.43328910,
            ]
        ),
        step_types=np.array([StepType.MID] * sum(lengths), dtype=StepType),
        env_infos={
            "success": np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=bool),
            "task_id": np.array([1] * 10 + [3] * 5 + [1] + [4]),
        },
        agent_infos={},
        episode_infos={},
        lengths=lengths,
    )

    log_file = tempfile.NamedTemporaryFile()
    csv_output = dowel.CsvOutput(log_file.name)
    logger.add_output(csv_output)
    log_multitask_performance(7, batch, 0.8, {1: "env1", 3: "env2", 4: "env3", 5: "env4"})
    logger.log(tabular)
    logger.dump_output_type(dowel.CsvOutput)
    with open(log_file.name, "r") as file:
        rows = list(csv.DictReader(file))
    res = {k: float(r) for (k, r) in rows[0].items()}
    assert res["env1/Iteration"] == 7
    assert res["env2/Iteration"] == 7
    assert res["env3/Iteration"] == 7
    assert res["env4/Iteration"] == 7
    assert res["env1/NumEpisodes"] == 2
    assert res["env2/NumEpisodes"] == 1
    assert res["env3/NumEpisodes"] == 1
    assert res["env4/NumEpisodes"] == 0
    assert math.isclose(res["env1/SuccessRate"], 0.5)
    assert math.isclose(res["env2/SuccessRate"], 1.0)
    assert math.isclose(res["env3/SuccessRate"], 1.0)
    assert math.isnan(res["env4/SuccessRate"])
    assert math.isnan(res["env4/AverageReturn"])
