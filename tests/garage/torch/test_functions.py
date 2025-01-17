"""Module to test garage.torch._functions."""
# yapf: disable
import collections

import numpy as np
import pytest
import torch
from torch import tensor
import torch.nn.functional as F

from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.torch import (as_torch_dict, compute_advantages,
                          flatten_to_single_vector, global_device, pad_to_last,
                          product_of_gaussians, set_gpu_mode, state_dict_to,
                          torch_to_np)
import garage.torch._functions as tu
from garage.torch.policies import DeterministicMLPPolicy

from tests.fixtures import TfGraphTestCase

# yapf: enable


def stack(d, arr):
    """Stack 'arr' 'd' times."""
    return np.repeat(np.expand_dims(arr, axis=0), repeats=d, axis=0)


ONES = np.ones((6,))
ZEROS = np.zeros((6,))
ARRANGE = np.arange(6)
PI_DIGITS = np.array([3, 1, 4, 1, 5, 9])
FIBS = np.array([1, 1, 2, 3, 5, 8])

nums_1d = np.arange(0, 4).astype(float)
nums_2d = np.arange(0, 4).astype(float).reshape(2, 2)
nums_3d = np.arange(0, 8).astype(float).reshape(2, 2, 2)


def test_utils_set_gpu_mode():
    """Test setting gpu mode to False to force CPU."""
    if torch.cuda.is_available():
        set_gpu_mode(mode=True)
        assert global_device() == torch.device("cuda:0")
        assert tu._USE_GPU
    else:
        set_gpu_mode(mode=False)
        assert global_device() == torch.device("cpu")
        assert not tu._USE_GPU
    assert not tu._GPU_ID


def test_torch_to_np():
    """Test whether tuples of tensors can be converted to np arrays."""
    tup = (torch.zeros(1), torch.zeros(1))
    np_out_1, np_out_2 = torch_to_np(tup)
    assert isinstance(np_out_1, np.ndarray)
    assert isinstance(np_out_2, np.ndarray)


def test_as_torch_dict():
    """Test if dict whose values are tensors can be converted to np arrays."""
    dic = {"a": np.zeros(1), "b": np.ones(1)}
    as_torch_dict(dic)
    for dic_value in dic.values():
        assert isinstance(dic_value, torch.Tensor)


def test_product_of_gaussians():
    """Test computing mu, sigma of product of gaussians."""
    size = 5
    mu = torch.ones(size)
    sigmas_squared = torch.ones(size)
    output = product_of_gaussians(mu, sigmas_squared)
    assert output[0] == 1
    assert output[1] == 1 / size


def test_flatten_to_single_vector():
    """Test test_flatten_to_single_vector"""
    x = torch.arange(12).view(2, 1, 3, 2)
    flatten_tensor = flatten_to_single_vector(x)
    expected = np.arange(12).reshape(2, 6)
    # expect [[ 0,  1,  2,  3,  4,  5], [ 6,  7,  8,  9, 10, 11]]
    assert torch.Size([2, 6]) == flatten_tensor.size()
    assert expected.shape == flatten_tensor.shape


@pytest.mark.gpu
def test_state_dict_to():
    """Test state_dict_to"""
    set_seed(42)
    # Using tensor instead of Tensor so it can be declared on GPU
    # pylint: disable=not-callable
    expected = collections.OrderedDict(
        [
            (
                "_module._layers.0.linear.weight",
                tensor(
                    [
                        [
                            0.13957974,
                            -0.2693157,
                            -0.19351028,
                            0.09471931,
                            -0.43573233,
                            0.03590716,
                            -0.4272097,
                            -0.13935488,
                            -0.35843086,
                            -0.25814268,
                            0.03060348,
                        ],
                        [
                            0.20623916,
                            -0.1914061,
                            0.46729338,
                            -0.5437773,
                            -0.50449526,
                            -0.55039907,
                            0.0141218,
                            -0.02489783,
                            0.26499796,
                            -0.03836302,
                            0.7235093,
                        ],
                    ],
                    device="cuda:0",
                ),
            ),
            ("_module._layers.0.linear.bias", tensor([0.0, 0.0], device="cuda:0")),
            (
                "_module._layers.1.linear.weight",
                tensor([[-0.7181905, -0.6284401], [0.10591025, -0.14771031]], device="cuda:0"),
            ),
            ("_module._layers.1.linear.bias", tensor([0.0, 0.0], device="cuda:0")),
            ("_module._output_layers.0.linear.weight", tensor([[-0.29133463, 0.58353233]], device="cuda:0")),
            ("_module._output_layers.0.linear.bias", tensor([0.0], device="cuda:0")),
        ]
    )
    # pylint: enable=not-callable
    env = normalize(GymEnv("InvertedDoublePendulum-v2"))
    policy = DeterministicMLPPolicy(
        env_spec=env.spec, hidden_sizes=[2, 2], hidden_nonlinearity=F.relu, output_nonlinearity=torch.tanh
    )
    moved_state_dict = state_dict_to(policy.state_dict(), "cuda")
    assert np.all([torch.allclose(expected[key], moved_state_dict[key]) for key in expected.keys()])
    assert np.all([moved_state_dict[key].is_cuda for key in moved_state_dict.keys()])


class TestTorchAlgoUtils(TfGraphTestCase):
    """Test class for torch algo utility functions."""

    # yapf: disable
    @pytest.mark.parametrize('discount', [1, 0.95])
    @pytest.mark.parametrize('num_eps', [1, 5])
    @pytest.mark.parametrize('gae_lambda', [0, 0.5, 1])
    @pytest.mark.parametrize('rewards_eps, baselines_eps', [
        (ONES, ZEROS),
        (PI_DIGITS, ARRANGE),
        (ONES, FIBS),
    ])
    # yapf: enable
    def test_compute_advantages(self, num_eps, discount, gae_lambda, rewards_eps, baselines_eps):
        """Test compute_advantage function."""

        def get_advantage(discount, gae_lambda, rewards, baselines):
            adv = torch.zeros(rewards.shape)
            for i in range(rewards.shape[0]):
                acc = 0
                for j in range(rewards.shape[1]):
                    acc = acc * discount * gae_lambda
                    acc += rewards[i][-j - 1] - baselines[i][-j - 1]
                    acc += discount * baselines[i][-j] if j else 0
                    adv[i][-j - 1] = acc
            return adv

        length = len(rewards_eps)

        rewards = torch.Tensor(stack(num_eps, rewards_eps))
        baselines = torch.Tensor(stack(num_eps, baselines_eps))
        expected_adv = get_advantage(discount, gae_lambda, rewards, baselines)
        computed_adv = compute_advantages(discount, gae_lambda, length, baselines, rewards)
        assert torch.allclose(expected_adv, computed_adv)

    def test_add_padding_last_1d(self):
        """Test pad_to_last function for 1d."""
        max_length = 10

        expected = F.pad(torch.Tensor(nums_1d), (0, max_length - nums_1d.shape[-1]))

        tensor_padding = pad_to_last(nums_1d, total_length=max_length)
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_1d, total_length=10, axis=0)
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_2d(self):
        """Test pad_to_last function for 2d."""
        max_length = 10

        tensor_padding = pad_to_last(nums_2d, total_length=10)
        expected = F.pad(torch.Tensor(nums_2d), (0, max_length - nums_2d.shape[-1]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_2d, total_length=10, axis=0)
        expected = F.pad(torch.Tensor(nums_2d), (0, 0, 0, max_length - nums_2d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_2d, total_length=10, axis=1)
        expected = F.pad(torch.Tensor(nums_2d), (0, max_length - nums_2d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

    def test_add_padding_last_3d(self):
        """Test pad_to_last function for 3d."""
        max_length = 10

        tensor_padding = pad_to_last(nums_3d, total_length=10)
        expected = F.pad(torch.Tensor(nums_3d), (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_3d, total_length=10, axis=0)
        expected = F.pad(torch.Tensor(nums_3d), (0, 0, 0, 0, 0, max_length - nums_3d.shape[0]))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_3d, total_length=10, axis=1)
        expected = F.pad(torch.Tensor(nums_3d), (0, 0, 0, max_length - nums_3d.shape[-1], 0, 0))
        assert expected.eq(tensor_padding).all()

        tensor_padding = pad_to_last(nums_3d, total_length=10, axis=2)
        expected = F.pad(torch.Tensor(nums_3d), (0, max_length - nums_3d.shape[-1], 0, 0, 0, 0))
        assert expected.eq(tensor_padding).all()

    @pytest.mark.parametrize("nums", [nums_1d, nums_2d, nums_3d])
    def test_out_of_index_error(self, nums):
        """Test pad_to_last raises IndexError."""
        with pytest.raises(IndexError):
            pad_to_last(nums, total_length=10, axis=len(nums.shape))


def test_expand_var():
    with pytest.raises(ValueError, match="test_var is length 2"):
        tu.expand_var("test_var", (1, 2), 3, "reference_var")


def test_value_at_axis():
    assert tu._value_at_axis("test_value", 0) == "test_value"
    assert tu._value_at_axis("test_value", 1) == "test_value"
    assert tu._value_at_axis(["a", "b", "c"], 0) == "a"
    assert tu._value_at_axis(["a", "b", "c"], 1) == "b"
    assert tu._value_at_axis(["a", "b", "c"], 2) == "c"
    assert tu._value_at_axis(("a", "b", "c"), 0) == "a"
    assert tu._value_at_axis(("a", "b", "c"), 1) == "b"
    assert tu._value_at_axis(("a", "b", "c"), 2) == "c"
    assert tu._value_at_axis(["test_value"], 3) == "test_value"
