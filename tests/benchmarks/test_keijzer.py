import pickle
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from finchge.benchmarks.regression.keijzer import (
    Keijzer1Benchmark,
    Keijzer2Benchmark,
    Keijzer3Benchmark,
    Keijzer4Benchmark,
    Keijzer5Benchmark,
    Keijzer6Benchmark,
    Keijzer7Benchmark,
    Keijzer8Benchmark,
    Keijzer9Benchmark,
    Keijzer10Benchmark,
    Keijzer11Benchmark,
    Keijzer12Benchmark,
    Keijzer13Benchmark,
    Keijzer14Benchmark,
    Keijzer15Benchmark,
    KeijzerFunction,
)


class TestKeijzerFunctionDefinitions:
    """Test that all function definitions are mathematically correct."""

    @pytest.mark.parametrize(
        "func_class,expected_name",
        [
            (Keijzer1Benchmark, "Keijzer-1"),
            (Keijzer2Benchmark, "Keijzer-2"),
            (Keijzer3Benchmark, "Keijzer-3"),
            (Keijzer4Benchmark, "Keijzer-4"),
            (Keijzer5Benchmark, "Keijzer-5"),
            (Keijzer6Benchmark, "Keijzer-6"),
            (Keijzer7Benchmark, "Keijzer-7"),
            (Keijzer8Benchmark, "Keijzer-8"),
            (Keijzer9Benchmark, "Keijzer-9"),
            (Keijzer10Benchmark, "Keijzer-10"),
            (Keijzer11Benchmark, "Keijzer-11"),
            (Keijzer12Benchmark, "Keijzer-12"),
            (Keijzer13Benchmark, "Keijzer-13"),
            (Keijzer14Benchmark, "Keijzer-14"),
            (Keijzer15Benchmark, "Keijzer-15"),
        ],
    )
    def test_benchmark_names(self, func_class, expected_name):
        bench = func_class(random_state=42)
        assert bench.metadata.name == expected_name

    def test_k1_function(self):
        bench = Keijzer1Benchmark()
        X = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        expected = 0.3 * X.flatten() * np.sin(2 * np.pi * X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K1]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k2_function(self):
        bench = Keijzer2Benchmark()
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        expected = 0.3 * X.flatten() * np.sin(2 * np.pi * X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K2]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k3_function(self):
        bench = Keijzer3Benchmark()
        X = np.array([[-3.0], [-1.5], [0.0], [1.5], [3.0]])
        expected = 0.3 * X.flatten() * np.sin(2 * np.pi * X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K3]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k4_function(self):
        bench = Keijzer4Benchmark()
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
        actual = bench._FUNCTIONS[KeijzerFunction.K4]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected, decimal=10)

    def test_k5_function(self):
        bench = Keijzer5Benchmark()
        X = np.array([[1.0], [1.5], [1.8], [2.0], [2.2]])

        def k5_manual(x):
            # Avoid division by zero at x=2
            with np.errstate(divide="ignore", invalid="ignore"):
                return 30 * (x - 1) * (x - 3) / ((x - 2) ** 2)

        expected = k5_manual(X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K5]["function"](X)

        # At x=2, function should be inf
        assert np.isinf(actual[3])
        # Other values should match
        assert_array_almost_equal(
            actual[[0, 1, 2, 4]].flatten(), expected[[0, 1, 2, 4]], decimal=10
        )

    def test_k6_function(self):
        bench = Keijzer6Benchmark()
        X = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        expected = X.flatten() + np.sin(X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K6]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k7_function(self):
        bench = Keijzer7Benchmark()
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        expected = np.log(X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K7]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k8_function(self):
        bench = Keijzer8Benchmark()
        X = np.array([[0.0], [1.0], [4.0], [9.0], [16.0]])
        expected = np.sqrt(X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K8]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k9_function(self):
        bench = Keijzer9Benchmark()
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        expected = np.arcsinh(X.flatten())
        actual = bench._FUNCTIONS[KeijzerFunction.K9]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k10_function(self):
        bench = Keijzer10Benchmark()
        X = np.array([[0.5, 2.0], [2.0, 3.0], [3.0, 2.0], [4.0, 0.5]])
        expected = X[:, 0] ** X[:, 1]
        actual = bench._FUNCTIONS[KeijzerFunction.K10]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k11_function(self):
        bench = Keijzer11Benchmark()
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = X[:, 0] * X[:, 1] + np.sin((X[:, 0] - 1) * (X[:, 1] - 1))
        actual = bench._FUNCTIONS[KeijzerFunction.K11]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k12_function(self):
        bench = Keijzer12Benchmark()
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = X[:, 0] ** 4 - X[:, 0] ** 3 + 0.5 * X[:, 1] ** 2 - X[:, 1]
        actual = bench._FUNCTIONS[KeijzerFunction.K12]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k13_function(self):
        bench = Keijzer13Benchmark()
        X = np.array([[0, 0], [np.pi / 2, 0], [0, np.pi / 2], [np.pi / 2, np.pi / 2]])
        expected = 6 * np.sin(X[:, 0]) * np.cos(X[:, 1])
        actual = bench._FUNCTIONS[KeijzerFunction.K13]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k14_function(self):
        bench = Keijzer14Benchmark()
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = 8 / (2 + X[:, 0] ** 2 + X[:, 1] ** 2)
        actual = bench._FUNCTIONS[KeijzerFunction.K14]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_k15_function(self):
        bench = Keijzer15Benchmark()
        X = np.array([[0, 0], [1, 1], [2, 2], [-1, -1]])
        expected = X[:, 0] ** 3 / 5 + X[:, 1] ** 3 / 2 - X[:, 1] - X[:, 0]
        actual = bench._FUNCTIONS[KeijzerFunction.K15]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)


class TestKeijzerDataGeneration:

    @pytest.mark.parametrize(
        "func_class,train_size,test_size",
        [
            # 1D functions with step sampling
            (
                Keijzer1Benchmark,
                21,
                201,
            ),  # -1 to 1 step 0.1: 21 train, step 0.01: 201 test
            (
                Keijzer2Benchmark,
                41,
                401,
            ),  # -2 to 2 step 0.1: 41 train, step 0.01: 401 test
            (
                Keijzer3Benchmark,
                61,
                601,
            ),  # -3 to 3 step 0.1: 61 train, step 0.01: 601 test
            (
                Keijzer4Benchmark,
                201,
                201,
            ),  # 0 to 10 step 0.05: 201 train, 0.05 to 10.05 step 0.05: 201 test
            (Keijzer5Benchmark, 40, 40),  # 0.05 to 2 step 0.05: 40 train, same for test
            (
                Keijzer6Benchmark,
                21,
                201,
            ),  # -1 to 1 step 0.1: 21 train, step 0.01: 201 test
            (
                Keijzer7Benchmark,
                100,
                991,
            ),  # 1 to 100 step 1: 100 train, step 0.1: 991 test
            (
                Keijzer8Benchmark,
                101,
                1001,
            ),  # 0 to 100 step 1: 101 train, step 0.1: 1001 test
            (
                Keijzer9Benchmark,
                101,
                1001,
            ),  # 0 to 100 step 1: 101 train, step 0.1: 1001 test
            # 2D functions with random sampling
            (Keijzer10Benchmark, 100, 1000),  # 2D random: 100 train, 1000 test
            (Keijzer11Benchmark, 100, 1000),
            (Keijzer12Benchmark, 100, 1000),
            (Keijzer13Benchmark, 100, 1000),
            (Keijzer14Benchmark, 100, 1000),
            (Keijzer15Benchmark, 100, 1000),
        ],
    )
    def test_data_shapes(self, func_class, train_size, test_size):
        """Test that data shapes match paper specifications."""
        bench = func_class(random_state=42)
        X_train, y_train, X_test, y_test = bench._generate_data()

        # Check shapes
        assert (
            X_train.shape[0] == train_size
        ), f"Train size mismatch: expected {train_size}, got {X_train.shape[0]}"
        assert y_train.shape[0] == train_size
        assert (
            X_test.shape[0] == test_size
        ), f"Test size mismatch: expected {test_size}, got {X_test.shape[0]}"
        assert y_test.shape[0] == test_size

        input_dim = bench.input_dim
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

    @pytest.mark.parametrize(
        "func_class",
        [
            Keijzer1Benchmark,
            Keijzer2Benchmark,
            Keijzer3Benchmark,
            Keijzer4Benchmark,
            Keijzer5Benchmark,
            Keijzer6Benchmark,
            Keijzer7Benchmark,
            Keijzer8Benchmark,
            Keijzer9Benchmark,
        ],
    )
    def test_step_sampling_1d(self, func_class):
        """Test that 1D functions use correct step sampling."""
        bench = func_class(random_state=42)
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

    @pytest.mark.parametrize(
        "func_class",
        [
            Keijzer10Benchmark,
            Keijzer11Benchmark,
            Keijzer12Benchmark,
            Keijzer13Benchmark,
            Keijzer14Benchmark,
            Keijzer15Benchmark,
        ],
    )
    def test_random_sampling_2d(self, func_class):
        """Test that 2D functions use appropriate sampling."""
        bench = func_class(random_state=42)
        X_train, _, X_test, _ = bench._generate_data()

        # Training should be random
        assert len(np.unique(X_train, axis=0)) > 0.9 * len(X_train)

        # Test should be grid-like
        low, high = bench.test_range
        n_per_dim = int(np.sqrt(bench.metadata.test_size))
        # expected_x = np.linspace(low, high, n_per_dim) # TODO Do we need this?

        # Check that test points are roughly grid-aligned
        unique_x1 = np.unique(np.round(X_test[:, 0], decimals=5))
        unique_x2 = np.unique(np.round(X_test[:, 1], decimals=5))

        assert len(unique_x1) == n_per_dim or len(unique_x1) == n_per_dim + 1
        assert len(unique_x2) == n_per_dim or len(unique_x2) == n_per_dim + 1


class TestKeijzerReproducibility:
    """Test reproducibility across instances and runs."""

    def test_reproducibility_across_instances(self):
        """Test that different instances with same seed produce same data."""
        bench1 = Keijzer4Benchmark(random_state=42)
        bench2 = Keijzer4Benchmark(random_state=42)

        X1_train, y1_train, X1_test, y1_test = bench1._generate_data()
        X2_train, y2_train, X2_test, y2_test = bench2._generate_data()

        assert_array_almost_equal(X1_train, X2_train)
        assert_array_almost_equal(y1_train, y2_train)
        assert_array_almost_equal(X1_test, X2_test)
        assert_array_almost_equal(y1_test, y2_test)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        bench1 = Keijzer10Benchmark(random_state=42)
        bench2 = Keijzer10Benchmark(random_state=24)

        X1_train, y1_train, _, _ = bench1._generate_data()
        X2_train, y2_train, _, _ = bench2._generate_data()

        assert not np.array_equal(X1_train, X2_train)
        assert not np.array_equal(y1_train, y2_train)

    def test_multiple_calls_produce_different_data(self):
        """Test that multiple calls to same instance produce different data."""
        bench = Keijzer10Benchmark(random_state=42)

        X1_train, y1_train, _, _ = bench._generate_data()
        X2_train, y2_train, _, _ = bench._generate_data()

        assert not np.array_equal(X1_train, X2_train)
        assert not np.array_equal(y1_train, y2_train)


class TestKeijzerPickling:
    """Test that benchmarks can be pickled and unpickled."""

    @pytest.mark.parametrize(
        "func_class",
        [
            Keijzer1Benchmark,
            Keijzer4Benchmark,
            Keijzer5Benchmark,
            Keijzer10Benchmark,
            Keijzer14Benchmark,
        ],
    )
    def test_pickle_roundtrip(self, func_class):
        """Test that benchmark can be pickled and unpickled."""
        bench = func_class(random_state=42)
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
    """Test edge cases and boundary conditions."""

    def test_k7_domain_restriction(self):
        """Test that K7 (log) is only evaluated at x > 0."""
        bench = Keijzer7Benchmark()
        X_train, _, _, _ = bench._generate_data()
        assert np.all(X_train > 0)

    def test_k8_domain_restriction(self):
        """Test that K8 (sqrt) is only evaluated at x >= 0."""
        bench = Keijzer8Benchmark()
        X_train, _, _, _ = bench._generate_data()
        assert np.all(X_train >= 0)

    def test_k10_domain_restriction(self):
        """Test that K10 (x^y) is evaluated in [0,1] domain."""
        bench = Keijzer10Benchmark(random_state=42)
        X_train, _, X_test, _ = bench._generate_data()

        assert np.all(
            (X_train >= 0) & (X_train <= 1)
        ), f"X_train values outside [0,1]: min={X_train.min()}, max={X_train.max()}"
        assert np.all(
            (X_test >= 0) & (X_test <= 1)
        ), f"X_test values outside [0,1]: min={X_test.min()}, max={X_test.max()}"

    def test_k4_complex_domain(self):
        """Test K4 over its domain."""
        bench = Keijzer4Benchmark()
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
    """Test that grammars are appropriate for each function."""

    def test_grammar_1d_vs_2d(self):
        """Test that 1D and 2D functions have different grammars."""
        bench_1d = Keijzer1Benchmark()
        bench_2d = Keijzer10Benchmark()

        grammar_1d = bench_1d.grammar()
        grammar_2d = bench_2d.grammar()

        assert "x0" in grammar_1d
        assert "x1" not in grammar_1d
        assert "x1" in grammar_2d or "y" in grammar_2d


class TestKeijzerScientificValidity:
    """Tests to ensure benchmarks are scientifically sound."""

    @pytest.mark.parametrize(
        "func_class,test_point,expected_value",
        [
            (Keijzer1Benchmark, np.array([[0.0]]), 0.0),
            (Keijzer1Benchmark, np.array([[0.25]]), 0.3 * 0.25 * np.sin(np.pi / 2)),
            (Keijzer2Benchmark, np.array([[0.0]]), 0.0),
            (Keijzer3Benchmark, np.array([[0.0]]), 0.0),
            (Keijzer4Benchmark, np.array([[0.0]]), 0.0),
            (Keijzer6Benchmark, np.array([[0.0]]), 0.0),
            (Keijzer7Benchmark, np.array([[1.0]]), 0.0),
            (Keijzer8Benchmark, np.array([[0.0]]), 0.0),
            (Keijzer9Benchmark, np.array([[0.0]]), 0.0),
            (Keijzer10Benchmark, np.array([[1.0, 2.0]]), 1.0),
            (Keijzer11Benchmark, np.array([[1.0, 1.0]]), 1.0 + np.sin(0)),
            (Keijzer12Benchmark, np.array([[0.0, 0.0]]), 0.0),
            (Keijzer13Benchmark, np.array([[0.0, 0.0]]), 0.0),
            (Keijzer14Benchmark, np.array([[0.0, 0.0]]), 4.0),
            (Keijzer15Benchmark, np.array([[0.0, 0.0]]), 0.0),
        ],
    )
    def test_known_values(self, func_class, test_point, expected_value):
        """Test functions at specific points."""
        bench = func_class()
        result = bench._FUNCTIONS[bench.function]["function"](test_point)
        assert_almost_equal(result.flatten()[0], expected_value, decimal=10)

    def test_k4_known_values(self):
        """Test K4 at specific points with correct values."""
        bench = Keijzer4Benchmark()

        test_points = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        # These are the CORRECT values from the mathematical function
        expected_values = np.array([-0.103268, 0.550654, 0.191505, -0.796949, 0.169341])

        actual = bench._FUNCTIONS[KeijzerFunction.K4]["function"](test_points).flatten()

        # Use appropriate tolerance
        np.testing.assert_allclose(actual, expected_values, rtol=1e-5, atol=1e-5)

    def test_k5_asymptotic_behavior(self):
        """Test asymptotic behavior of K5."""
        bench = Keijzer5Benchmark()

        # Test points approaching asymptote
        test_points = [1.99, 1.999, 2.001, 2.01]
        # expected_pattern = []  # TODO DO WE NEED THIS ?

        for x in test_points:
            X = np.array([[x]])
            y = bench._FUNCTIONS[KeijzerFunction.K5]["function"](X).flatten()[0]

            # Should be negative
            assert y < 0, f"At x={x}, expected negative, got {y}"

            # Should be large in magnitude
            magnitude = abs(y)
            expected_magnitude = 30 * abs((x - 1) * (x - 3)) / ((x - 2) ** 2)
            assert (
                abs(magnitude - expected_magnitude) < 1
            ), f"At x={x}, expected magnitude ~{expected_magnitude:.0f}, got {magnitude:.0f}"

        # Test symmetry
        y_left = bench._FUNCTIONS[KeijzerFunction.K5]["function"](
            np.array([[1.99]])
        ).flatten()[0]
        y_right = bench._FUNCTIONS[KeijzerFunction.K5]["function"](
            np.array([[2.01]])
        ).flatten()[0]

        # Values should be approximately equal (both negative)
        assert (
            abs(y_left - y_right) < abs(y_left) * 0.01
        ), f"Left and right values should be similar: {y_left} vs {y_right}"
