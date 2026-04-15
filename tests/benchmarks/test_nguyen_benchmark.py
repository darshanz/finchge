import hashlib
import pickle
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from finchge.benchmarks.regression import NguyenBenchmark


class TestNguyenFunctionDefinitions:
    @pytest.mark.parametrize(
        "version,name",
        [
            (1, "Nguyen-1"),
            (2, "Nguyen-2"),
            (3, "Nguyen-3"),
            (4, "Nguyen-4"),
            (5, "Nguyen-5"),
            (6, "Nguyen-6"),
            (7, "Nguyen-7"),
            (8, "Nguyen-8"),
            (9, "Nguyen-9"),
            (10, "Nguyen-10"),
        ],
    )
    def test_benchmark_names(self, version, name):
        bench = NguyenBenchmark(version=version, random_state=42)
        assert bench.metadata.name == name

    def test_function_1_cubic(self):
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
        expected = np.array([-6.0, -1.0, 0.0, 3.0, 14.0])
        bench = NguyenBenchmark(version=1)
        actual = bench.func(X)
        actual_flat = actual.flatten()
        assert_array_almost_equal(actual_flat, expected)

    def test_function_2_quartic(self):
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        expected = np.array(
            [16 - 8 + 4 - 2, 1 - 1 + 1 - 1, 0, 1 + 1 + 1 + 1, 16 + 8 + 4 + 2]
        )
        bench = NguyenBenchmark(version=2)
        actual = bench.func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_3_quintic(self):
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
        bench = NguyenBenchmark(version=3)
        actual = bench.func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_4_sextic(self):
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
        bench = NguyenBenchmark(version=4)
        actual = bench.func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_5_trig(self):
        X = np.array([[0.0], [np.pi / 2], [np.pi]])
        expected = np.array(
            [
                np.sin(0) * np.cos(0) - 1,  # -1
                np.sin((np.pi / 2) ** 2) * np.cos(np.pi / 2) - 1,  # 0 - 1 = -1
                np.sin(np.pi**2) * np.cos(np.pi) - 1,  # sin(π²) * (-1) - 1
            ]
        )
        bench = NguyenBenchmark(version=5)
        actual = bench.func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_6_composite_trig(self):
        X = np.array([[0.0], [1.0], [2.0]])
        expected = np.array(
            [
                np.sin(0) + np.sin(0 + 0),  # 0
                np.sin(1) + np.sin(1 + 1),  # sin(1) + sin(2)
                np.sin(2) + np.sin(2 + 4),  # sin(2) + sin(6)
            ]
        )
        bench = NguyenBenchmark(version=6)
        actual = bench.func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_7_log(self):
        X = np.array([[0.0], [1.0], [2.0]])
        expected = np.array(
            [
                np.log(1) + np.log(1),  # 0 + 0 = 0
                np.log(2) + np.log(2),  # log(2) + log(2)
                np.log(3) + np.log(5),  # log(3) + log(5)
            ]
        )
        bench = NguyenBenchmark(version=7)
        actual = bench.func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_8_sqrt(self):
        X = np.array([[0.0], [1.0], [4.0], [9.0]])
        expected = np.array([0, 1, 2, 3])
        bench = NguyenBenchmark(version=8)
        actual = bench.func(X)
        assert_array_almost_equal(actual.flatten(), expected)

    def test_function_9_2d_trig(self):
        X = np.array([[0, 0], [np.pi / 2, 1], [np.pi, 2]])
        expected = np.array(
            [
                np.sin(0) + np.sin(0**2),  # 0
                np.sin(np.pi / 2) + np.sin(1**2),  # 1 + sin(1)
                np.sin(np.pi) + np.sin(2**2),  # 0 + sin(4)
            ]
        )
        bench = NguyenBenchmark(version=9)
        actual = bench.func(X)
        assert_array_almost_equal(actual, expected)

    def test_function_10_2d_product(self):
        X = np.array([[0, 0], [np.pi / 2, 0], [np.pi / 2, np.pi / 2]])
        expected = np.array(
            [
                2 * np.sin(0) * np.cos(0),  # 0
                2 * np.sin(np.pi / 2) * np.cos(0),  # 2 * 1 * 1 = 2
                2 * np.sin(np.pi / 2) * np.cos(np.pi / 2),  # 2 * 1 * 0 = 0
            ]
        )
        bench = NguyenBenchmark(version=10)
        actual = bench.func(X)
        assert_array_almost_equal(actual, expected)


class TestNguyenBenchmarkInitialization:
    def test_custom_parameters(self):
        bench = NguyenBenchmark(
            version=1,
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
            NguyenBenchmark(version=1, train_samples=0)

        with pytest.raises(ValueError, match="test_samples must be positive"):
            NguyenBenchmark(version=1, test_samples=-10)

        with pytest.raises(ValueError, match="train_type must be one of"):
            NguyenBenchmark(version=1, train_type="invalid")

        with pytest.raises(ValueError, match="Invalid range"):
            NguyenBenchmark(version=1, x_range=(5, -5))

    def test_invalid_function(self):
        with pytest.raises(ValueError, match="not found"):
            NguyenBenchmark(13)

        with pytest.raises(ValueError, match="not found"):
            NguyenBenchmark("invalid")

    def test_2d_range_handling(self):
        """Test that 2D ranges are handled correctly."""
        # Single range for both dimensions
        bench = NguyenBenchmark(version=9, x_range=(0, 2))
        assert bench.x_range == ((0, 2), (0, 2))

        # Explicit 2D range
        bench = NguyenBenchmark(version=9, x_range=((0, 1), (-1, 1)))
        assert bench.x_range == ((0, 1), (-1, 1))

        # Invalid 2D range
        with pytest.raises(ValueError):
            bench = NguyenBenchmark(version=9, x_range=((1, 0), (0, 1)))  # low > high


class TestNguyenDataGeneration:
    def test_reproducibility(self):
        """Test that same random state produces identical data."""
        bench1 = NguyenBenchmark(version=1, random_state=42)
        bench2 = NguyenBenchmark(version=1, random_state=42)

        X1_train, y1_train, X1_test, y1_test = bench1._generate_data()
        X2_train, y2_train, X2_test, y2_test = bench2._generate_data()

        assert_array_almost_equal(X1_train, X2_train)
        assert_array_almost_equal(y1_train, y2_train)
        assert_array_almost_equal(X1_test, X2_test)
        assert_array_almost_equal(y1_test, y2_test)

    def test_different_random_states(self):
        bench1 = NguyenBenchmark(version=1, random_state=42, train_type="uniform")
        bench2 = NguyenBenchmark(version=1, random_state=2, train_type="uniform")

        X1_train, y1_train, _, _ = bench1._generate_data()
        X2_train, y2_train, _, _ = bench2._generate_data()

        # They should be different
        assert not np.array_equal(X1_train, X2_train)

    @pytest.mark.parametrize(
        "version,sample_type",
        [
            (1, "uniform"),
            (1, "grid"),
            (9, "uniform"),
            (9, "grid"),
        ],
    )
    def test_sample_types(self, version, sample_type):
        bench = NguyenBenchmark(
            version=version,
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
        if bench.dim == 1:
            low, high = bench.x_range
            assert np.all((X_train >= low) & (X_train <= high))
        else:
            (low1, high1), (low2, high2) = bench.x_range
            assert np.all((X_train[:, 0] >= low1) & (X_train[:, 0] <= high1))
            assert np.all((X_train[:, 1] >= low2) & (X_train[:, 1] <= high2))

    def test_grid_sampling_2d(self):
        bench = NguyenBenchmark(version=9, test_samples=49, test_type="grid")
        _, _, X_test, _ = bench._generate_data()
        assert len(X_test) == 49
        unique_x1 = len(np.unique(X_test[:, 0]))
        unique_x2 = len(np.unique(X_test[:, 1]))
        assert unique_x1 * unique_x2 >= 49

    @pytest.mark.parametrize(
        "version",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    def test_data_shapes(self, version):
        """Test that data shapes are correct for all functions."""
        bench = NguyenBenchmark(version=version, train_samples=36, test_samples=900)
        X_train, y_train, X_test, y_test = bench._generate_data()

        input_dim = bench.dim

        assert X_train.shape == (36, input_dim)
        assert y_train.shape == (36,)
        assert X_test.shape == (900, input_dim)
        assert y_test.shape == (900,)

    @pytest.mark.parametrize(
        "version,test_point,expected",
        [
            (1, np.array([[2.0]]), 14.0),
            (5, np.array([[0.0]]), -1.0),
            (7, np.array([[1.0]]), 2 * np.log(2)),
            (9, np.array([[0.5, 0.5]]), np.sin(0.5) + np.sin(0.25)),
        ],
    )
    def test_known_values(self, version, test_point, expected):
        bench = NguyenBenchmark(version=version)
        result = bench.func(test_point)
        assert_almost_equal(result[0], expected, decimal=10)


class TestNguyenGrammars:
    def test_grammar_1d_vs_2d(self):
        grammar_1d = NguyenBenchmark(version=1).grammar()
        grammar_2d = NguyenBenchmark(version=9).grammar()

        assert "x0" in grammar_1d.terminals
        assert "x1" not in grammar_1d.terminals
        assert "x1" in grammar_2d.terminals

    @pytest.mark.parametrize(
        "version,restricted",
        [
            (1, False),
            (2, False),
            (3, False),
            (4, False),
            (5, False),
            (6, False),
            (7, True),  # log requires positive domain
            (8, True),  # sqrt requires non-negative domain
            (9, False),
            (10, False),
        ],
    )
    def test_restricted_domain_grammars(self, version, restricted):
        """Test that functions with restricted domains have appropriate grammars."""
        bench = NguyenBenchmark(version=version)
        grammar = bench.grammar_str()

        if restricted:
            assert "pdiv" in grammar
        else:
            # Should have standard functions
            assert "sin" in grammar
            assert "cos" in grammar


class TestNguyenMetadata:
    @pytest.mark.parametrize(
        "version,input_dim,output_dim",
        [
            (1, 1, 1),
            (2, 1, 1),
            (3, 1, 1),
            (4, 1, 1),
            (5, 1, 1),
            (6, 1, 1),
            (7, 1, 1),
            (8, 1, 1),
            (9, 2, 1),
            (10, 2, 1),
        ],
    )
    def test_dimensions(self, version, input_dim, output_dim):
        bench = NguyenBenchmark(version=version)
        assert bench.metadata.input_dim == input_dim
        assert bench.metadata.output_dim == output_dim

    def test_metadata_fields(self):
        bench = NguyenBenchmark(version=1)
        meta = bench.metadata

        assert meta.name == "Nguyen-1"
        assert meta.category == "regression"
        assert meta.train_size == 20
        assert meta.test_size == 1000


class TestNguyenSerialization:
    def test_pickle_roundtrip(self):
        bench = NguyenBenchmark(version=5, random_state=42, train_samples=30)
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
        bench1 = NguyenBenchmark(version=3, random_state=123)
        X1_train, y1_train, X1_test, y1_test = bench1._generate_data()

        # second run with same seed 123
        bench2 = NguyenBenchmark(version=3, random_state=123)
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
        bench = NguyenBenchmark(version=1, random_state=42)
        X_train, y_train, X_test, y_test = bench._generate_data()

        # Create hashes of the data
        train_hash = hashlib.md5(X_train.tobytes()).hexdigest()
        test_hash = hashlib.md5(X_test.tobytes()).hexdigest()

        # These hashes should remain constant across versions
        assert train_hash == "7521d4c94772c9030e44097bf3fe7626"
        assert test_hash == "416cd7529cde1d55a30e33b9973d6e9a"


class TestNguyenEdgeCases:
    """
    Test edge cases and boundary conditions.
    """

    def test_single_sample(self):
        """Test with single sample."""
        bench = NguyenBenchmark(version=1, train_samples=1, test_samples=1)
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert X_train.shape == (1, 1)
        assert y_train.shape == (1,)
        assert X_test.shape == (1, 1)
        assert y_test.shape == (1,)

    def test_extreme_ranges(self):
        bench = NguyenBenchmark(version=1, x_range=(-1e6, 1e6))
        X_train, y_train, _, _ = bench._generate_data()

        assert np.all((X_train >= -1e6) & (X_train <= 1e6))

    def test_domain_boundaries(self):
        # Nguyen-7 (log) at lower bound
        bench = NguyenBenchmark(version=7, x_range=(0, 2))
        X = np.array([[0.0]])  # Should be valid (log(1) = 0)
        y = bench.func(X)
        assert not np.isnan(y)
        assert not np.isinf(y)

        # Nguyen-8 (sqrt) at lower bound
        bench = NguyenBenchmark(version=8, x_range=(0, 4))
        X = np.array([[0.0]])
        y = bench.func(X)
        assert y[0] == 0.0


class TestNguyenScientificValidity:
    def test_function_continuity_random_paths(self):
        rng = np.random.default_rng(42)

        for func_ in range(10):
            func = func_ + 1
            if func in [7, 8]:
                continue

            bench = NguyenBenchmark(func)

            for _ in range(10):  # Test 10 random paths
                if bench.dim == 1:
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

                y = bench.func(X)

                # Check that function values are finite
                assert np.all(
                    np.isfinite(y)
                ), f"Function {bench.name} produced non-finite values"

                # Check for unrealistic jumps
                diffs = np.abs(np.diff(y.flatten()))
                max_jump = np.max(diffs)
                value_range = np.max(y) - np.min(y)

                if value_range > 0:
                    # No single jump should be more than half the total range
                    assert (
                        max_jump < 0.5 * value_range
                    ), f"Function {bench.name} has suspiciously large jump of {max_jump} (range={value_range})"

    def test_function_range(self):
        for func in range(10):
            bench = NguyenBenchmark(func + 1)
            X_train, y_train, X_test, y_test = bench._generate_data()

            # Check for NaN or Inf
            assert not np.any(np.isnan(y_train))
            assert not np.any(np.isinf(y_train))
            assert not np.any(np.isnan(y_test))
            assert not np.any(np.isinf(y_test))

    def test_training_size_adequacy(self):
        """Test that training size (20) is appropriate for symbolic regression."""
        for func in range(10):
            bench = NguyenBenchmark(func + 1, train_samples=20)
            X_train, y_train, _, _ = bench._generate_data()

            # Check that we have enough unique points
            if bench.dim == 1:
                unique_points = len(np.unique(X_train))
                assert unique_points >= min(15, 20)  # At least 15 unique points

    def test_test_density(self):
        """Test that test set (1000 points) provides good coverage."""
        for func in [1, 9]:  # Sample 1D and 2D
            bench = NguyenBenchmark(func, test_samples=1000, test_type="grid")
            _, _, X_test, _ = bench._generate_data()

            if bench.dim == 1:
                # Check that points are well-distributed
                intervals = np.diff(np.sort(X_test.flatten()))
                assert np.std(intervals) < 0.01  # Low variance in spacing
            else:
                # For 2D, check that we have a reasonable grid
                unique_x1 = len(np.unique(X_test[:, 0]))
                unique_x2 = len(np.unique(X_test[:, 1]))
                assert unique_x1 * unique_x2 >= 900  # Close to 1000
