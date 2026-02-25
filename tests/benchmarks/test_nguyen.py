import hashlib
import pickle
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from finchge.benchmarks.regression.nguyen import (
    Nguyen1Benchmark,
    Nguyen2Benchmark,
    Nguyen3Benchmark,
    Nguyen4Benchmark,
    Nguyen5Benchmark,
    Nguyen6Benchmark,
    Nguyen7Benchmark,
    Nguyen8Benchmark,
    Nguyen9Benchmark,
    Nguyen10Benchmark,
    Nguyen11Benchmark,
    Nguyen12Benchmark,
    NguyenBenchmark,
    NguyenFunction,
)
from finchge.grammar import Grammar


class TestNguyenFunctionDefinitions:
    """Test that all function definitions are mathematically correct."""

    @pytest.mark.parametrize(
        "func_class,expected_name",
        [
            (Nguyen1Benchmark, "Nguyen-1"),
            (Nguyen2Benchmark, "Nguyen-2"),
            (Nguyen3Benchmark, "Nguyen-3"),
            (Nguyen4Benchmark, "Nguyen-4"),
            (Nguyen5Benchmark, "Nguyen-5"),
            (Nguyen6Benchmark, "Nguyen-6"),
            (Nguyen7Benchmark, "Nguyen-7"),
            (Nguyen8Benchmark, "Nguyen-8"),
            (Nguyen9Benchmark, "Nguyen-9"),
            (Nguyen10Benchmark, "Nguyen-10"),
            (Nguyen11Benchmark, "Nguyen-11"),
            (Nguyen12Benchmark, "Nguyen-12"),
        ],
    )
    def test_benchmark_names(self, func_class, expected_name):
        """Test that benchmarks have correct names."""
        bench = func_class(random_state=42)
        assert bench.metadata.name == expected_name

    def test_function_1_cubic(self):
        """Test Nguyen-1: x^3 + x^2 + x"""
        bench = Nguyen1Benchmark()
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        expected = np.array([-6, -1, 0, 3, 14])
        actual = bench._FUNCTIONS[NguyenFunction.N1]["function"](X)
        # Flatten the actual values for comparison
        actual_flat = actual.flatten()
        assert_array_almost_equal(actual_flat, expected)

    def test_function_2_quartic(self):
        """Test Nguyen-2: x^4 + x^3 + x^2 + x"""
        bench = Nguyen2Benchmark()
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        expected = np.array(
            [16 - 8 + 4 - 2, 1 - 1 + 1 - 1, 0, 1 + 1 + 1 + 1, 16 + 8 + 4 + 2]
        )
        # expected: [10, 0, 0, 4, 30]
        actual = bench._FUNCTIONS[NguyenFunction.N2]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_3_quintic(self):
        """Test Nguyen-3: x^5 + x^4 + x^3 + x^2 + x"""
        bench = Nguyen3Benchmark()
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        expected = np.array(
            [
                -32 + 16 - 8 + 4 - 2,
                -1 + 1 - 1 + 1 - 1,
                0,
                1 + 1 + 1 + 1 + 1,
                32 + 16 + 8 + 4 + 2,
            ]
        )
        # expected: [-22, -1, 0, 5, 62]
        actual = bench._FUNCTIONS[NguyenFunction.N3]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_4_sextic(self):
        """Test Nguyen-4: x^6 + x^5 + x^4 + x^3 + x^2 + x"""
        bench = Nguyen4Benchmark()
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        expected = np.array(
            [
                64 - 32 + 16 - 8 + 4 - 2,
                1 - 1 + 1 - 1 + 1 - 1,
                0,
                1 + 1 + 1 + 1 + 1 + 1,
                64 + 32 + 16 + 8 + 4 + 2,
            ]
        )
        # expected: [42, 0, 0, 6, 126]
        actual = bench._FUNCTIONS[NguyenFunction.N4]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_5_trig(self):
        """Test Nguyen-5: sin(x^2) * cos(x) - 1"""
        bench = Nguyen5Benchmark()
        X = np.array([[0.0], [np.pi / 2], [np.pi]])
        expected = np.array(
            [
                np.sin(0) * np.cos(0) - 1,  # -1
                np.sin((np.pi / 2) ** 2) * np.cos(np.pi / 2) - 1,  # 0 - 1 = -1
                np.sin(np.pi**2) * np.cos(np.pi) - 1,  # sin(π²) * (-1) - 1
            ]
        )
        actual = bench._FUNCTIONS[NguyenFunction.N5]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_6_composite_trig(self):
        """Test Nguyen-6: sin(x) + sin(x + x^2)"""
        bench = Nguyen6Benchmark()
        X = np.array([[0.0], [1.0], [2.0]])
        expected = np.array(
            [
                np.sin(0) + np.sin(0 + 0),  # 0
                np.sin(1) + np.sin(1 + 1),  # sin(1) + sin(2)
                np.sin(2) + np.sin(2 + 4),  # sin(2) + sin(6)
            ]
        )
        actual = bench._FUNCTIONS[NguyenFunction.N6]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_7_log(self):
        """Test Nguyen-7: log(x + 1) + log(x^2 + 1)"""
        bench = Nguyen7Benchmark()
        X = np.array([[0.0], [1.0], [2.0]])
        expected = np.array(
            [
                np.log(1) + np.log(1),  # 0 + 0 = 0
                np.log(2) + np.log(2),  # log(2) + log(2)
                np.log(3) + np.log(5),  # log(3) + log(5)
            ]
        )
        actual = bench._FUNCTIONS[NguyenFunction.N7]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_8_sqrt(self):
        """Test Nguyen-8: sqrt(x)"""
        bench = Nguyen8Benchmark()
        X = np.array([[0.0], [1.0], [4.0], [9.0]])
        expected = np.array([0, 1, 2, 3])
        actual = bench._FUNCTIONS[NguyenFunction.N8]["function"](X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_9_2d_trig(self):
        """Test Nguyen-9: sin(x) + sin(y^2)"""
        bench = Nguyen9Benchmark()
        X = np.array([[0, 0], [np.pi / 2, 1], [np.pi, 2]])
        expected = np.array(
            [
                np.sin(0) + np.sin(0**2),  # 0
                np.sin(np.pi / 2) + np.sin(1**2),  # 1 + sin(1)
                np.sin(np.pi) + np.sin(2**2),  # 0 + sin(4)
            ]
        )
        actual = bench._FUNCTIONS[NguyenFunction.N9]["function"](X)
        assert_array_almost_equal(actual, expected)

    def test_function_10_2d_product(self):
        """Test Nguyen-10: 2*sin(x)*cos(y)"""
        bench = Nguyen10Benchmark()
        X = np.array([[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]])
        expected = np.array(
            [
                2 * np.sin(0) * np.cos(0),  # 0
                2 * np.sin(np.pi / 2) * np.cos(0),  # 2 * 1 * 1 = 2
                2 * np.sin(np.pi / 2) * np.cos(np.pi / 2),  # 2 * 1 * 0 = 0
            ]
        )
        actual = bench._FUNCTIONS[NguyenFunction.N10]["function"](X)
        assert_array_almost_equal(actual, expected)

    def test_function_11_power(self):
        """Test Nguyen-11: x^y"""
        bench = Nguyen11Benchmark()
        X = np.array([[2, 3], [4, 2], [3, 3]])
        expected = np.array([8, 16, 27])
        actual = bench._FUNCTIONS[NguyenFunction.N11]["function"](X)
        assert_array_almost_equal(actual, expected)

    def test_function_12_2d_poly(self):
        """Test Nguyen-12: x^4 - x^3 + 0.5*y^2 - y"""
        bench = Nguyen12Benchmark()
        X = np.array([[0, 0], [1, 1], [2, 2]])
        expected = np.array(
            [
                0 - 0 + 0 - 0,  # 0
                1 - 1 + 0.5 * 1 - 1,  # -0.5
                16 - 8 + 0.5 * 4 - 2,  # 8 + 2 - 2 = 8
            ]
        )
        actual = bench._FUNCTIONS[NguyenFunction.N12]["function"](X)
        assert_array_almost_equal(actual, expected)


class TestNguyenBenchmarkInitialization:
    """Test benchmark initialization and parameter handling."""

    def test_default_initialization(self):
        """Test default parameters for each function."""
        for func in NguyenFunction:
            bench = NguyenBenchmark(func)

            assert bench.train_samples == 20
            assert bench.test_samples == 1000
            assert bench.train_type == "uniform"
            assert bench.test_type == "grid"

            # For 2D functions, the range gets converted to tuple of tuples
            if bench.func_info["input_dim"] == 1:
                assert bench.x_range == bench.func_info["default_range"]
            else:
                # For 2D, default_range is a single tuple, but x_range is converted
                default = bench.func_info["default_range"]
                expected_range = ((default[0], default[1]), (default[0], default[1]))
                assert bench.x_range == expected_range

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        bench = Nguyen1Benchmark(
            random_state=123,
            train_samples=50,
            test_samples=500,
            x_range=(-5, 5),
            train_type="grid",
            test_type="uniform",
        )

        assert bench.random_state == 123
        assert bench.train_samples == 50
        assert bench.test_samples == 500
        assert bench.x_range == (-5, 5)
        assert bench.train_type == "grid"
        assert bench.test_type == "uniform"

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="train_samples must be positive"):
            Nguyen1Benchmark(train_samples=0)

        with pytest.raises(ValueError, match="test_samples must be positive"):
            Nguyen1Benchmark(test_samples=-10)

        with pytest.raises(ValueError, match="train_type must be one of"):
            Nguyen1Benchmark(train_type="invalid")

        with pytest.raises(ValueError, match="Invalid range"):
            Nguyen1Benchmark(x_range=(5, -5))

    def test_function_parsing(self):
        """Test different ways to specify the function."""
        bench1 = NguyenBenchmark(1)
        bench2 = NguyenBenchmark("N1")
        bench3 = NguyenBenchmark("Nguyen-1")
        bench4 = NguyenBenchmark("nguyen1")
        bench5 = NguyenBenchmark(NguyenFunction.N1)

        assert bench1.function == NguyenFunction.N1
        assert bench2.function == NguyenFunction.N1
        assert bench3.function == NguyenFunction.N1
        assert bench4.function == NguyenFunction.N1
        assert bench5.function == NguyenFunction.N1

    def test_invalid_function(self):
        """Test that invalid function identifiers raise errors."""
        with pytest.raises(
            ValueError, match="Nguyen function number must be between 1-12"
        ):
            NguyenBenchmark(13)

        with pytest.raises(ValueError, match="Invalid Nguyen function"):
            NguyenBenchmark("invalid")

    def test_2d_range_handling(self):
        """Test that 2D ranges are handled correctly."""
        # Single range for both dimensions
        bench = Nguyen9Benchmark(x_range=(0, 2))
        assert bench.x_range == ((0, 2), (0, 2))

        # Explicit 2D range
        bench = Nguyen9Benchmark(x_range=((0, 1), (-1, 1)))
        assert bench.x_range == ((0, 1), (-1, 1))

        # Invalid 2D range
        with pytest.raises(ValueError):
            bench = Nguyen9Benchmark(x_range=((1, 0), (0, 1)))  # low > high


class TestNguyenDataGeneration:
    """Test data generation for all functions."""

    def test_reproducibility(self):
        """Test that same random state produces identical data."""
        bench1 = Nguyen1Benchmark(random_state=42)
        bench2 = Nguyen1Benchmark(random_state=42)

        X1_train, y1_train, X1_test, y1_test = bench1._generate_data()
        X2_train, y2_train, X2_test, y2_test = bench2._generate_data()

        assert_array_almost_equal(X1_train, X2_train)
        assert_array_almost_equal(y1_train, y2_train)
        assert_array_almost_equal(X1_test, X2_test)
        assert_array_almost_equal(y1_test, y2_test)

    def test_different_random_states(self):
        """Test that different random states produce different data."""
        bench1 = Nguyen1Benchmark(random_state=42)
        bench2 = Nguyen1Benchmark(random_state=24)

        X1_train, y1_train, _, _ = bench1._generate_data()
        X2_train, y2_train, _, _ = bench2._generate_data()

        # They should be different (very low probability of being equal)
        assert not np.array_equal(X1_train, X2_train)

    @pytest.mark.parametrize(
        "func_class,sample_type",
        [
            (Nguyen1Benchmark, "uniform"),
            (Nguyen1Benchmark, "grid"),
            (Nguyen9Benchmark, "uniform"),
            (Nguyen9Benchmark, "grid"),
        ],
    )
    def test_sample_types(self, func_class, sample_type):
        """Test different sampling methods."""
        bench = func_class(
            random_state=42,
            train_samples=100,
            train_type=sample_type,
            test_type=sample_type,
        )

        X_train, y_train, X_test, y_test = bench._generate_data()

        assert len(X_train) == 100
        assert len(y_train) == 100
        assert len(X_test) == 1000
        assert len(y_test) == 1000

        # Check that values are within range
        if bench.func_info["input_dim"] == 1:
            low, high = bench.x_range
            assert np.all((X_train >= low) & (X_train <= high))
        else:
            (low1, high1), (low2, high2) = bench.x_range
            assert np.all((X_train[:, 0] >= low1) & (X_train[:, 0] <= high1))
            assert np.all((X_train[:, 1] >= low2) & (X_train[:, 1] <= high2))

    def test_grid_sampling_2d(self):
        """Test that 2D grid sampling produces expected number of points."""
        bench = Nguyen9Benchmark(test_samples=50, test_type="grid")
        _, _, X_test, _ = bench._generate_data()

        # Should have exactly 50 points (not necessarily a perfect square)
        assert len(X_test) == 50

        # Points should be roughly grid-like (unique combinations)
        unique_x1 = len(np.unique(X_test[:, 0]))
        unique_x2 = len(np.unique(X_test[:, 1]))
        assert unique_x1 * unique_x2 >= 50

    @pytest.mark.parametrize(
        "func_class",
        [
            Nguyen1Benchmark,
            Nguyen2Benchmark,
            Nguyen3Benchmark,
            Nguyen4Benchmark,
            Nguyen5Benchmark,
            Nguyen6Benchmark,
            Nguyen7Benchmark,
            Nguyen8Benchmark,
            Nguyen9Benchmark,
            Nguyen10Benchmark,
            Nguyen11Benchmark,
            Nguyen12Benchmark,
        ],
    )
    def test_data_shapes(self, func_class):
        """Test that data shapes are correct for all functions."""
        bench = func_class(train_samples=30, test_samples=200)
        X_train, y_train, X_test, y_test = bench._generate_data()

        input_dim = bench.func_info["input_dim"]

        assert X_train.shape == (30, input_dim)
        assert y_train.shape == (30,)
        assert X_test.shape == (200, input_dim)
        assert y_test.shape == (200,)

    @pytest.mark.parametrize(
        "func_class,test_point,expected",
        [
            (Nguyen1Benchmark, np.array([[2.0]]), 14.0),
            (Nguyen5Benchmark, np.array([[0.0]]), -1.0),
            (Nguyen7Benchmark, np.array([[1.0]]), 2 * np.log(2)),
            (Nguyen9Benchmark, np.array([[0.5, 0.5]]), np.sin(0.5) + np.sin(0.25)),
        ],
    )
    def test_known_values(self, func_class, test_point, expected):
        """Test that functions produce known values at specific points."""
        bench = func_class()
        result = bench._FUNCTIONS[bench.function]["function"](test_point)
        assert_almost_equal(result[0], expected, decimal=10)


class TestNguyenGrammars:
    """Test that grammars are appropriate for each function."""

    def test_grammar_1d_vs_2d(self):
        """Test that 1D and 2D functions have different grammars."""
        bench_1d = Nguyen1Benchmark()
        bench_2d = Nguyen9Benchmark()

        grammar_1d = Grammar(bench_1d.grammar())
        grammar_2d = Grammar(bench_2d.grammar())

        assert "x0" in grammar_1d.terminals
        assert "x1" not in grammar_1d.terminals
        assert "x1" in grammar_2d.terminals

    @pytest.mark.parametrize(
        "func_class,restricted",
        [
            (Nguyen1Benchmark, False),
            (Nguyen2Benchmark, False),
            (Nguyen3Benchmark, False),
            (Nguyen4Benchmark, False),
            (Nguyen5Benchmark, False),
            (Nguyen6Benchmark, False),
            (Nguyen7Benchmark, True),  # log requires positive domain
            (Nguyen8Benchmark, True),  # sqrt requires non-negative domain
            (Nguyen9Benchmark, False),
            (Nguyen10Benchmark, False),
            (Nguyen11Benchmark, False),
            (Nguyen12Benchmark, False),
        ],
    )
    def test_restricted_domain_grammars(self, func_class, restricted):
        """Test that functions with restricted domains have appropriate grammars."""
        bench = func_class()
        grammar = bench.grammar()

        if restricted:
            # Should mention protected functions or have extra constants
            assert "const" in grammar
        else:
            # Should have standard functions
            assert "sin" in grammar
            assert "cos" in grammar


class TestNguyenMetadata:
    """Test metadata for all functions."""

    @pytest.mark.parametrize(
        "func_class,input_dim,output_dim",
        [
            (Nguyen1Benchmark, 1, 1),
            (Nguyen2Benchmark, 1, 1),
            (Nguyen3Benchmark, 1, 1),
            (Nguyen4Benchmark, 1, 1),
            (Nguyen5Benchmark, 1, 1),
            (Nguyen6Benchmark, 1, 1),
            (Nguyen7Benchmark, 1, 1),
            (Nguyen8Benchmark, 1, 1),
            (Nguyen9Benchmark, 2, 1),
            (Nguyen10Benchmark, 2, 1),
            (Nguyen11Benchmark, 2, 1),
            (Nguyen12Benchmark, 2, 1),
        ],
    )
    def test_dimensions(self, func_class, input_dim, output_dim):
        """Test that input/output dimensions are correct."""
        bench = func_class()
        assert bench.metadata.input_dim == input_dim
        assert bench.metadata.output_dim == output_dim

    def test_metadata_fields(self):
        """Test that all metadata fields are present."""
        bench = Nguyen1Benchmark()
        meta = bench.metadata

        assert meta.name == "Nguyen-1"
        assert meta.category == "regression"
        assert meta.description is not None
        assert meta.reference is not None
        assert meta.train_size == 20
        assert meta.test_size == 1000


class TestNguyenSerialization:
    """Test that benchmarks can be serialized (important for distributed computing)."""

    def test_pickle_roundtrip(self):
        """Test that benchmark can be pickled and unpickled."""
        bench = Nguyen5Benchmark(random_state=42, train_samples=30)
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


class TestNguyenReproducibility:
    """Test that results are reproducible across runs and platforms."""

    def test_deterministic_output(self):
        """Test that multiple calls produce same output."""

        # first run with seed 123
        bench1 = Nguyen3Benchmark(random_state=123)
        X1_train, y1_train, X1_test, y1_test = bench1._generate_data()

        # second run with same seed 123
        bench2 = Nguyen3Benchmark(random_state=123)
        X2_train, y2_train, X2_test, y2_test = bench2._generate_data()

        # should be same
        assert_array_almost_equal(X1_train, X2_train)
        assert_array_almost_equal(y1_train, y2_train)
        assert_array_almost_equal(X1_test, X2_test)
        assert_array_almost_equal(y1_test, y2_test)

    def test_hash_consistency(self):
        """
        Test that data hashes are consistent (for regression testing).
        """
        bench = Nguyen1Benchmark(random_state=42)
        X_train, y_train, X_test, y_test = bench._generate_data()

        # Create hashes of the data
        train_hash = hashlib.md5(X_train.tobytes()).hexdigest()
        test_hash = hashlib.md5(X_test.tobytes()).hexdigest()

        # These hashes should remain constant across versions
        assert train_hash == "820abe090ed7aca299d9529be00de1fb"
        assert test_hash == "6e5a2e389c3e1f67cf1cbcac6037340c"


class TestNguyenEdgeCases:
    """
    Test edge cases and boundary conditions.
    """

    def test_single_sample(self):
        """Test with single sample."""
        bench = Nguyen1Benchmark(train_samples=1, test_samples=1)
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert X_train.shape == (1, 1)
        assert y_train.shape == (1,)
        assert X_test.shape == (1, 1)
        assert y_test.shape == (1,)

    def test_extreme_ranges(self):
        """
        Test with extreme input ranges.
        """
        bench = Nguyen1Benchmark(x_range=(-1e6, 1e6))
        X_train, y_train, _, _ = bench._generate_data()

        assert np.all((X_train >= -1e6) & (X_train <= 1e6))

    def test_domain_boundaries(self):
        """Test functions at domain boundaries."""
        # Nguyen-7 (log) at lower bound
        bench = Nguyen7Benchmark(x_range=(0, 2))
        X = np.array([[0.0]])  # Should be valid (log(1) = 0)
        y = bench._FUNCTIONS[bench.function]["function"](X)
        assert not np.isnan(y)
        assert not np.isinf(y)

        # Nguyen-8 (sqrt) at lower bound
        bench = Nguyen8Benchmark(x_range=(0, 4))
        X = np.array([[0.0]])
        y = bench._FUNCTIONS[bench.function]["function"](X)
        assert y[0] == 0.0


class TestNguyenScientificValidity:
    """Tests to ensure benchmarks are scientifically sound."""

    def test_function_continuity_random_paths(self):
        """Test continuity along random paths through the domain."""
        rng = np.random.default_rng(42)

        for func in NguyenFunction:
            if func in [NguyenFunction.N7, NguyenFunction.N8]:
                continue

            bench = NguyenBenchmark(func)

            for _ in range(10):  # Test 10 random paths
                if bench.func_info["input_dim"] == 1:
                    low, high = bench.x_range
                    # Create a random path by sorting random points
                    X = rng.uniform(low, high, (100, 1))
                    X = np.sort(X, axis=0)
                else:
                    (low1, high1), (low2, high2) = bench.x_range
                    # Create a random 2D path
                    t = np.linspace(0, 1, 100)
                    # Use smooth random functions for the path
                    x1 = low1 + (high1 - low1) * (
                        0.5 + 0.5 * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
                    )
                    x2 = low2 + (high2 - low2) * (
                        0.5 + 0.5 * np.cos(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
                    )
                    X = np.column_stack([x1, x2])

                y = bench._FUNCTIONS[func]["function"](X)

                # Check that function values are finite
                assert np.all(
                    np.isfinite(y)
                ), f"Function {bench.func_info['name']} produced non-finite values"

                # Check for unrealistic jumps
                diffs = np.abs(np.diff(y.flatten()))
                max_jump = np.max(diffs)
                value_range = np.max(y) - np.min(y)

                if value_range > 0:
                    # No single jump should be more than half the total range
                    assert (
                        max_jump < 0.5 * value_range
                    ), f"Function {bench.func_info['name']} has suspiciously large jump of {max_jump} (range={value_range})"

    def test_function_range(self):
        """Test that functions produce values in expected ranges."""
        for func in NguyenFunction:
            bench = NguyenBenchmark(func)
            X_train, y_train, X_test, y_test = bench._generate_data()

            # Check for NaN or Inf
            assert not np.any(np.isnan(y_train))
            assert not np.any(np.isinf(y_train))
            assert not np.any(np.isnan(y_test))
            assert not np.any(np.isinf(y_test))

    def test_training_size_adequacy(self):
        """Test that training size (20) is appropriate for symbolic regression."""
        for func in NguyenFunction:
            bench = NguyenBenchmark(func, train_samples=20)
            X_train, y_train, _, _ = bench._generate_data()

            # Check that we have enough unique points
            if bench.func_info["input_dim"] == 1:
                unique_points = len(np.unique(X_train))
                assert unique_points >= min(15, 20)  # At least 15 unique points

    def test_test_density(self):
        """Test that test set (1000 points) provides good coverage."""
        for func in [NguyenFunction.N1, NguyenFunction.N9]:  # Sample 1D and 2D
            bench = NguyenBenchmark(func, test_samples=1000, test_type="grid")
            _, _, X_test, _ = bench._generate_data()

            if bench.func_info["input_dim"] == 1:
                # Check that points are well-distributed
                intervals = np.diff(np.sort(X_test.flatten()))
                assert np.std(intervals) < 0.01  # Low variance in spacing
            else:
                # For 2D, check that we have a reasonable grid
                unique_x1 = len(np.unique(X_test[:, 0]))
                unique_x2 = len(np.unique(X_test[:, 1]))
                assert unique_x1 * unique_x2 >= 900  # Close to 1000
