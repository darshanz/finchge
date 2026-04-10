import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from finchge.benchmarks.regression import KeijzerBenchmark

# Load keys once for the test module
SPEC_PATH = (
    Path(__file__).parent.parent.parent
    / "finchge/benchmarks/regression/keijzer/functions.json"
)
with open(SPEC_PATH) as f:
    KEIJZER_VERSIONS = list(json.load(f).keys())


class TestKeijzerFunctionDefinitions:
    """Test that all function definitions are mathematically correct."""

    @pytest.mark.parametrize("version", KEIJZER_VERSIONS)
    def test_all_versions(self, version):
        v_int = int(version)
        bench = KeijzerBenchmark(version=v_int)
        assert bench.metadata.name == f"Keijzer-{version}"

    def test_k1_function(self):
        X = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        expected = 0.3 * X.flatten() * np.sin(2 * np.pi * X.flatten())
        actual = KeijzerBenchmark(1).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k2_function(self):
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        expected = 0.3 * X.flatten() * np.sin(2 * np.pi * X.flatten())
        actual = KeijzerBenchmark(2).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k3_function(self):
        X = np.array([[-3.0], [-1.5], [0.0], [1.5], [3.0]])
        expected = 0.3 * X.flatten() * np.sin(2 * np.pi * X.flatten())
        actual = KeijzerBenchmark(3).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k4_function(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])

        def k4_manual(x):
            return (
                x**3
                * np.exp(-x)
                * np.cos(x)
                * np.sin(x)
                * (np.sin(x) ** 2 * np.cos(x) - 1)
            )

        expected = k4_manual(X.flatten())
        actual = KeijzerBenchmark(4).func(X)
        assert_array_almost_equal(actual.flatten(), expected, decimal=10)

    def test_k5_function(self):
        X = np.array([[1.0], [1.5], [1.8], [2.0], [2.2]])

        def k5_manual(x):
            # Avoid division by zero at x=2
            with np.errstate(divide="ignore", invalid="ignore"):
                return 30 * (x - 1) * (x - 3) / ((x - 2) ** 2)

        expected = k5_manual(X.flatten())
        actual = KeijzerBenchmark(5).func(X)

        # At x=2, function should be inf
        assert np.isinf(actual[3])
        # Other values should match
        assert_array_almost_equal(
            actual[[0, 1, 2, 4]].flatten(), expected[[0, 1, 2, 4]], decimal=10
        )

    def test_k6_function(self):
        X = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        expected = X.flatten() + np.sin(X.flatten())
        actual = KeijzerBenchmark(6).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k7_function(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        expected = np.log(X.flatten())
        actual = KeijzerBenchmark(7).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k8_function(self):
        X = np.array([[0.0], [1.0], [4.0], [9.0], [16.0]])
        expected = np.sqrt(X.flatten())
        actual = KeijzerBenchmark(8).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k9_function(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        expected = np.arcsinh(X.flatten())
        actual = KeijzerBenchmark(9).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k10_function(self):
        X = np.array([[0.5, 2.0], [2.0, 3.0], [3.0, 2.0], [4.0, 0.5]])
        expected = X[:, 0] ** X[:, 1]
        actual = KeijzerBenchmark(10).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k11_function(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = X[:, 0] * X[:, 1] + np.sin((X[:, 0] - 1) * (X[:, 1] - 1))
        actual = KeijzerBenchmark(11).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k12_function(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = X[:, 0] ** 4 - X[:, 0] ** 3 + 0.5 * X[:, 1] ** 2 - X[:, 1]
        actual = KeijzerBenchmark(12).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k13_function(self):
        X = np.array([[0, 0], [np.pi / 2, 0], [0, np.pi / 2], [np.pi / 2, np.pi / 2]])
        expected = 6 * np.sin(X[:, 0]) * np.cos(X[:, 1])
        actual = KeijzerBenchmark(13).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k14_function(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = 8 / (2 + X[:, 0] ** 2 + X[:, 1] ** 2)
        actual = KeijzerBenchmark(14).func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k15_function(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = X[:, 0] ** 3 / 5 + X[:, 1] ** 3 / 2 - X[:, 1] - X[:, 0]
        actual = KeijzerBenchmark(15).func(X)
        assert_array_almost_equal(actual.flatten(), expected)


class TestKeijzerDataGeneration:
    @pytest.mark.parametrize("version", range(1, 16))
    def test_keijzer_sample_sizes(self, version):
        def calc_keijzer_size(r, step, default):
            if step is not None:
                return int(round((r[1] - r[0]) / step)) + 1
            return default

        bench = KeijzerBenchmark(version=version)
        expected_train = calc_keijzer_size(bench.train_range, bench.train_step, 100)
        expected_test = calc_keijzer_size(bench.test_range, bench.test_step, 1000)
        assert bench.train_size == expected_train
        assert bench.test_size == expected_test

    @pytest.mark.parametrize("version", range(1, 16))
    def test_data_shapes(self, version):
        bench = KeijzerBenchmark(version=version)
        X_train, y_train, X_test, y_test = bench._generate_data()

        # Check shapes
        assert (
            X_train.shape[0] == bench.train_size
        ), f"Train size mismatch: expected {bench.train_size}, got {X_train.shape[0]}"
        assert y_train.shape[0] == bench.train_size
        assert (
            X_test.shape[0] == bench.test_size
        ), f"Test size mismatch: expected {bench.test_size}, got {X_test.shape[0]}"
        assert y_test.shape[0] == bench.test_size

        input_dim = bench.dim
        assert X_train.shape[1] == input_dim
        assert X_test.shape[1] == input_dim

        # For step-sampled functions, verify the step size
        if hasattr(bench, "train_step") and bench.train_step is not None:
            # Check that points are evenly spaced
            if input_dim == 1:
                diffs = np.diff(np.sort(X_train.flatten()))
                expected_step = bench.train_step
                assert np.all(
                    np.abs(diffs - expected_step) < 1e-10
                ), f"Steps not uniform: {diffs[:5]}"

    @pytest.mark.parametrize("version", [a for a in range(1, 10)])  # K1-K9
    def test_step_sampling_1d(self, version):
        bench = KeijzerBenchmark(version=version, random_state=42)
        X_train, _, _, _ = bench._generate_data()

        # Check that points are evenly spaced
        diffs = np.diff(np.sort(X_train.flatten()))
        assert np.all(np.abs(diffs - diffs[0]) < 1e-10)

        # Check that points match step specification
        low, high = bench.train_range
        step = bench.train_step
        expected_points = np.arange(low, high + step / 2, step)
        assert len(X_train) == len(expected_points)
        assert_array_almost_equal(np.sort(X_train.flatten()), expected_points)

    @pytest.mark.parametrize("version", [a for a in range(10, 16)])  # K10-K15
    def test_random_sampling_2d(self, version):
        bench = KeijzerBenchmark(version=version, random_state=42)
        X_train, _, X_test, _ = bench._generate_data()

        # Training should be random
        assert len(np.unique(X_train, axis=0)) > 0.9 * len(X_train)

        # Test should be grid-like
        low, high = bench.test_range
        n_per_dim = int(np.sqrt(bench.metadata.test_size))
        unique_x1 = np.unique(np.round(X_test[:, 0], decimals=5))
        unique_x2 = np.unique(np.round(X_test[:, 1], decimals=5))

        assert len(unique_x1) == n_per_dim or len(unique_x1) == n_per_dim + 1
        assert len(unique_x2) == n_per_dim or len(unique_x2) == n_per_dim + 1


class TestKeijzerReproducibility:
    def test_reproducibility_across_instances(self):
        bench1 = KeijzerBenchmark(version=4, random_state=42)
        bench2 = KeijzerBenchmark(version=4, random_state=42)

        X1_train, y1_train, X1_test, y1_test = bench1._generate_data()
        X2_train, y2_train, X2_test, y2_test = bench2._generate_data()

        assert_array_almost_equal(X1_train, X2_train)
        assert_array_almost_equal(y1_train, y2_train)
        assert_array_almost_equal(X1_test, X2_test)
        assert_array_almost_equal(y1_test, y2_test)

    def test_different_seeds_produce_different_data(self):
        bench1 = KeijzerBenchmark(version=10, random_state=42)
        bench2 = KeijzerBenchmark(version=10, random_state=24)

        X1_train, y1_train, _, _ = bench1._generate_data()
        X2_train, y2_train, _, _ = bench2._generate_data()

        assert not np.array_equal(X1_train, X2_train)
        assert not np.array_equal(y1_train, y2_train)

    def test_multiple_calls_produce_different_data(self):
        bench = KeijzerBenchmark(version=10, random_state=42)

        X1_train, y1_train, _, _ = bench._generate_data()
        X2_train, y2_train, _, _ = bench._generate_data()

        assert not np.array_equal(X1_train, X2_train)
        assert not np.array_equal(y1_train, y2_train)


class TestKeijzerPickling:
    @pytest.mark.parametrize("version", [1, 4, 5, 10, 14])
    def test_pickle_roundtrip(self, version):
        bench = KeijzerBenchmark(version=version, random_state=42)
        X_train_original, y_train_original, _, _ = bench._generate_data()

        # Pickle and unpickle
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(bench, f)
            f.flush()
            f.seek(0)
            bench_loaded = pickle.load(f)

        # Check that loaded benchmark produces same data
        X_train_loaded, y_train_loaded, _, _ = bench_loaded._generate_data()

        assert_array_almost_equal(X_train_original, X_train_loaded)
        assert_array_almost_equal(y_train_original, y_train_loaded)

        # Check that metadata is preserved
        assert bench_loaded.metadata.name == bench.metadata.name
        assert bench_loaded.metadata.train_size == bench.metadata.train_size


class TestKeijzerEdgeCases:
    def test_k7_domain_restriction(self):
        bench = KeijzerBenchmark(version=7, random_state=42)
        X_train, _, _, _ = bench._generate_data()
        assert np.all(X_train > 0)

    def test_k8_domain_restriction(self):
        bench = KeijzerBenchmark(version=8, random_state=42)
        X_train, _, _, _ = bench._generate_data()
        assert np.all(X_train >= 0)

    def test_k10_domain_restriction(self):
        bench = KeijzerBenchmark(version=10, random_state=42)
        X_train, _, X_test, _ = bench._generate_data()

        assert np.all(
            (X_train >= 0) & (X_train <= 1)
        ), f"X_train values outside [0,1]: min={X_train.min()}, max={X_train.max()}"
        assert np.all(
            (X_test >= 0) & (X_test <= 1)
        ), f"X_test values outside [0,1]: min={X_test.min()}, max={X_test.max()}"

    def test_k4_complex_domain(self):
        bench = KeijzerBenchmark(version=4, random_state=42)
        X_train, y_train, _, _ = bench._generate_data()

        # Values should be finite
        assert np.all(np.isfinite(y_train))

        # Known property: function oscillates and decays
        # X_vals = X_train.flatten() # TODO , usable??
        y_vals = y_train.flatten()

        # Should have multiple zero crossings
        sign_changes = np.sum(np.diff(np.sign(y_vals)) != 0)
        assert sign_changes > 5


class TestKeijzerGrammars:
    def test_grammar_1d_vs_2d(self):
        bench_1d = KeijzerBenchmark(version=1, random_state=42)
        bench_2d = KeijzerBenchmark(version=10, random_state=42)

        grammar_1d = bench_1d.grammar_str()
        grammar_2d = bench_2d.grammar_str()

        assert "x0" in grammar_1d
        assert "x1" not in grammar_1d
        assert "x1" in grammar_2d or "y" in grammar_2d


class TestKeijzerScientificValidity:
    @pytest.mark.parametrize(
        "version,test_point,expected_value",
        [
            (1, np.array([[0.0]]), 0.0),
            (1, np.array([[0.25]]), 0.3 * 0.25 * np.sin(np.pi / 2)),
            (2, np.array([[0.0]]), 0.0),
            (3, np.array([[0.0]]), 0.0),
            (4, np.array([[0.0]]), 0.0),
            (6, np.array([[0.0]]), 0.0),
            (7, np.array([[1.0]]), 0.0),
            (8, np.array([[0.0]]), 0.0),
            (9, np.array([[0.0]]), 0.0),
            (10, np.array([[1.0, 2.0]]), 1.0),
            (11, np.array([[1.0, 1.0]]), 1.0 + np.sin(0)),
            (12, np.array([[0.0, 0.0]]), 0.0),
            (13, np.array([[0.0, 0.0]]), 0.0),
            (14, np.array([[0.0, 0.0]]), 4.0),
            (15, np.array([[0.0, 0.0]]), 0.0),
        ],
    )
    def test_known_values(self, version, test_point, expected_value):
        bench = KeijzerBenchmark(version=version, random_state=42)
        result = bench.func(test_point)
        assert_almost_equal(result.flatten()[0], expected_value, decimal=10)

    def test_k4_known_values(self):
        test_points = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        expected_values = np.array([-0.103268, 0.550654, 0.191505, -0.796949, 0.169341])
        bench = KeijzerBenchmark(version=4, random_state=42)
        actual = bench.func(test_points)
        np.testing.assert_allclose(actual, expected_values, rtol=1e-5, atol=1e-5)

    def test_k5_asymptotic_behavior(self):
        bench = KeijzerBenchmark(version=5, random_state=42)
        test_points = [1.99, 1.999, 2.001, 2.01]
        for x in test_points:
            X = np.array([[x]])
            y = bench.func(X).flatten()[0]
            assert y < 0, f"At x={x}, expected negative, got {y}"

            # Should be large in magnitude
            magnitude = abs(y)
            expected_magnitude = 30 * abs((x - 1) * (x - 3)) / ((x - 2) ** 2)
            assert (
                abs(magnitude - expected_magnitude) < 1
            ), f"At x={x}, expected magnitude ~{expected_magnitude:.0f}, got {magnitude:.0f}"

        # Test symmetry
        y_left = bench.func(np.array([[1.99]])).flatten()[0]
        y_right = bench.func(np.array([[2.01]])).flatten()[0]

        # Values should be approximately equal (both negative)
        assert (
            abs(y_left - y_right) < abs(y_left) * 0.01
        ), f"Left and right values should be similar: {y_left} vs {y_right}"
