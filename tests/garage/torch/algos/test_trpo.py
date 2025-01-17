"""This script creates a test that fails when TRPO performance is too low."""
import pytest
import torch

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.sampler import LocalSampler
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config


class TestTRPO:
    """Test class for TRPO."""

    def setup_method(self):
        """Setup method which is called before every test."""
        self.env = normalize(GymEnv("InvertedDoublePendulum-v2", max_episode_length=100))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.value_function = GaussianMLPValueFunction(env_spec=self.env.spec)

    def teardown_method(self):
        """Teardown method which is called after every test."""
        self.env.close()

    @pytest.mark.mujoco
    def test_trpo_pendulum(self):
        """Test TRPO with Pendulum environment."""
        deterministic.set_seed(0)
        sampler = LocalSampler(agents=self.policy, envs=self.env, max_episode_length=self.env.spec.max_episode_length)
        trainer = Trainer(snapshot_config)
        algo = TRPO(
            env_spec=self.env.spec,
            policy=self.policy,
            value_function=self.value_function,
            sampler=sampler,
            discount=0.99,
            gae_lambda=0.98,
        )

        trainer.setup(algo, self.env)
        last_avg_ret = trainer.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0
