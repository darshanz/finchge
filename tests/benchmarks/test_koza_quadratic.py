import hashlib
import pickle
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from finchge.benchmarks.regression.koza_quartic import KozaQuarticBenchmark


class TestKozaQuarticFunction:
    """Test the mathematical correctness of the Koza Quartic function."""

    def test_function_values(self):
        """Test function at known points."""
        bench = KozaQuarticBenchmark()

        # Test points
        test_inputs = np.array([[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]])

        # Expected values: x^4 + x^3 + x^2 + x
        # Calculate manually to verify
        expected = []
        for x in test_inputs.flatten():
            val = x**4 + x**3 + x**2 + x
            expected.append(val)
        expected = np.array(expected)

        # Get actual values
        actual = bench._quartic_function(test_inputs)

        assert_array_almost_equal(actual, expected, decimal=10)

    def test_function_at_boundaries(self):
        """Test function at range boundaries."""
        bench = KozaQuarticBenchmark(x_range=(-2, 2))

        # Test at boundaries
        X = np.array([[-2.0], [2.0]])
        y = bench._quartic_function(X)

        # Expected: (-2)^4 + (-2)^3 + (-2)^2 + (-2) = 16 - 8 + 4 - 2 = 10
        # Expected: 2^4 + 2^3 + 2^2 + 2 = 16 + 8 + 4 + 2 = 30
        expected = np.array([10.0, 30.0])

        assert_array_almost_equal(y, expected)

    def test_function_symmetry(self):
        """Test that function is not symmetric (should be different for x and -x except at 0)."""
        bench = KozaQuarticBenchmark()

        X = np.array([[0.5], [-0.5]])
        y = bench._quartic_function(X)

        # Values should be different
        assert y[0] != y[1]

        # For small x, the odd terms dominate the asymmetry
        assert y[0] > y[1]  # 0.5 gives positive, -0.5 gives negative


class TestKozaQuarticInitialization:
    """Test benchmark initialization and parameter handling."""

    def test_default_initialization(self):
        """Test default parameters."""
        bench = KozaQuarticBenchmark()

        assert bench.train_samples == 100
        assert bench.test_samples == 1000
        assert bench.x_range == (-1, 1)
        assert bench.train_type == "uniform"
        assert bench.test_type == "grid"

        # Test metadata
        assert bench.metadata.name == "Koza Quartic"
        assert bench.metadata.input_dim == 1
        assert bench.metadata.output_dim == 1
        assert bench.metadata.train_size == 100
        assert bench.metadata.test_size == 1000

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        bench = KozaQuarticBenchmark(
            random_state=42,
            train_samples=50,
            test_samples=500,
            x_range=(-2, 2),
            train_type="grid",
            test_type="uniform",
        )

        assert bench.random_state == 42
        assert bench.train_samples == 50
        assert bench.test_samples == 500
        assert bench.x_range == (-2, 2)
        assert bench.train_type == "grid"
        assert bench.test_type == "uniform"

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="train_samples must be positive"):
            KozaQuarticBenchmark(train_samples=0)

        with pytest.raises(ValueError, match="train_samples must be positive"):
            KozaQuarticBenchmark(train_samples=-10)

        with pytest.raises(ValueError, match="test_samples must be positive"):
            KozaQuarticBenchmark(test_samples=0)

        with pytest.raises(ValueError, match="Invalid range"):
            KozaQuarticBenchmark(x_range=(5, -5))

        with pytest.raises(ValueError, match="train_type must be one of"):
            KozaQuarticBenchmark(train_type="invalid")

        with pytest.raises(ValueError, match="test_type must be one of"):
            KozaQuarticBenchmark(test_type="invalid")


class TestKozaQuarticDataGeneration:
    """Test data generation for Koza Quartic."""

    def test_reproducibility(self):
        """Test that same random state produces identical data."""
        bench1 = KozaQuarticBenchmark(random_state=42)
        bench2 = KozaQuarticBenchmark(random_state=42)

        X1_train, y1_train, X1_test, y1_test = bench1._generate_data()
        X2_train, y2_train, X2_test, y2_test = bench2._generate_data()

        assert_array_almost_equal(X1_train, X2_train)
        assert_array_almost_equal(y1_train, y2_train)
        assert_array_almost_equal(X1_test, X2_test)
        assert_array_almost_equal(y1_test, y2_test)

    def test_different_random_states(self):
        """Test that different random states produce different data."""
        bench1 = KozaQuarticBenchmark(random_state=42)
        bench2 = KozaQuarticBenchmark(random_state=24)

        X1_train, y1_train, _, _ = bench1._generate_data()
        X2_train, y2_train, _, _ = bench2._generate_data()

        # They should be different (very low probability of being equal)
        assert not np.array_equal(X1_train, X2_train)

    @pytest.mark.parametrize("sample_type", ["uniform", "grid", "random"])
    def test_sample_types_train(self, sample_type):
        """Test different training sampling methods."""
        bench = KozaQuarticBenchmark(
            random_state=42, train_samples=50, train_type=sample_type
        )

        X_train, y_train, _, _ = bench._generate_data()

        assert len(X_train) == 50
        assert len(y_train) == 50

        # Check that values are within range
        low, high = bench.x_range
        assert np.all((X_train >= low) & (X_train <= high))

        # For grid sampling, check that points are evenly spaced
        if sample_type == "grid":
            intervals = np.diff(np.sort(X_train.flatten()))
            assert np.all(np.abs(intervals - intervals[0]) < 1e-10)

    @pytest.mark.parametrize("sample_type", ["uniform", "grid", "random"])
    def test_sample_types_test(self, sample_type):
        """Test different test sampling methods."""
        bench = KozaQuarticBenchmark(
            random_state=42, test_samples=100, test_type=sample_type
        )

        _, _, X_test, y_test = bench._generate_data()

        assert len(X_test) == 100
        assert len(y_test) == 100

        # Check that values are within range
        low, high = bench.x_range
        assert np.all((X_test >= low) & (X_test <= high))

    def test_data_shapes(self):
        """Test that data shapes are correct."""
        bench = KozaQuarticBenchmark(train_samples=30, test_samples=200)
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert X_train.shape == (30, 1)
        assert y_train.shape == (30,)
        assert X_test.shape == (200, 1)
        assert y_test.shape == (200,)

    def test_data_consistency(self):
        """Test that y values are correctly computed from X."""
        bench = KozaQuarticBenchmark(random_state=42)
        X_train, y_train, X_test, y_test = bench._generate_data()

        # Recompute y from X
        y_train_computed = bench._quartic_function(X_train)
        y_test_computed = bench._quartic_function(X_test)

        assert_array_almost_equal(y_train, y_train_computed)
        assert_array_almost_equal(y_test, y_test_computed)

    def test_train_test_independence(self):
        """Test that train and test sets are independent."""
        bench = KozaQuarticBenchmark(random_state=42)
        X_train, _, X_test, _ = bench._generate_data()

        # For random sampling, there should be no exact duplicates
        # (very low probability of overlap with these sample sizes)
        train_set = set(tuple(x) for x in X_train)
        test_set = set(tuple(x) for x in X_test)

        # Allow for possible overlap due to grid sampling
        if bench.train_type != "grid" and bench.test_type != "grid":
            assert len(train_set.intersection(test_set)) == 0


class TestKozaQuarticGrammar:
    """Test the grammar for Koza Quartic."""

    def test_grammar_content(self):
        """Test that grammar contains expected elements."""
        bench = KozaQuarticBenchmark()
        grammar = bench.grammar_str()

        # Should contain basic operators
        assert "+" in grammar
        assert "-" in grammar
        assert "*" in grammar
        assert "/" in grammar

        # Should contain functions
        assert "sin" in grammar
        assert "cos" in grammar
        assert "exp" in grammar
        assert "log" in grammar
        assert "sqrt" in grammar

        # Should contain variable
        assert "x" in grammar or "x[0]" in grammar or "X" in grammar

        # Should contain constants
        assert "const" in grammar.lower()

    def test_grammar_validity(self):
        """Test that grammar is syntactically valid (basic check)."""
        bench = KozaQuarticBenchmark()
        grammar = bench.grammar_str()

        # Check for proper BNF structure
        assert "::=" in grammar
        assert "<expr>" in grammar

        # Check that all non-terminals are defined
        lines = grammar.strip().split("\n")
        defined_nonterminals = set()
        used_nonterminals = set()

        for line in lines:
            line = line.strip()
            if "::=" in line:
                nonterminal = line.split("::=")[0].strip()
                defined_nonterminals.add(nonterminal)
                # Extract used nonterminals from RHS
                rhs = line.split("::=")[1]
                for token in rhs.split():
                    if token.startswith("<") and token.endswith(">"):
                        used_nonterminals.add(token)

        # All used nonterminals should be defined
        # (except maybe built-in ones)
        for used in used_nonterminals:
            if used not in ["<expr>", "<op>", "<func>", "<var>", "<const>"]:
                assert used in defined_nonterminals


class TestKozaQuarticMetadata:
    """Test metadata for Koza Quartic."""

    def test_metadata_fields(self):
        """Test that all metadata fields are present and correct."""
        bench = KozaQuarticBenchmark(train_samples=50, test_samples=200)
        meta = bench.metadata

        assert meta.name == "Koza Quartic"
        assert meta.category == "regression"
        assert "quartic polynomial" in meta.description.lower()
        assert "Koza" in meta.reference
        assert meta.input_dim == 1
        assert meta.output_dim == 1
        assert meta.train_size == 50
        assert meta.test_size == 200

    def test_metadata_update_with_parameters(self):
        """Test that metadata updates with custom parameters."""
        bench = KozaQuarticBenchmark(train_samples=75, test_samples=150)

        assert bench.metadata.train_size == 75
        assert bench.metadata.test_size == 150


class TestKozaQuarticSerialization:
    """
    Test that benchmark can be serialized.
    Uses rng which cannot be pickled :; need to reset later
    """

    def test_pickle_roundtrip(self):
        """Test that benchmark can be pickled and unpickled."""
        bench = KozaQuarticBenchmark(random_state=42, train_samples=30)
        X_train_original, y_train_original, _, _ = bench._generate_data()

        # Pickle and unpickle
        with tempfile.NamedTemporaryFile() as f:
            # Store the random state before pickling
            random_state = bench.random_state

            # Remove the unpicklable _rng attribute
            if hasattr(bench, "_rng"):
                delattr(bench, "_rng")

            pickle.dump(bench, f)
            f.flush()
            f.seek(0)
            bench_loaded = pickle.load(f)

            # Restore the random state and recreate _rng
            bench_loaded.random_state = random_state
            bench_loaded._rng = np.random.default_rng(random_state)

        # Check that loaded benchmark produces same data
        X_train_loaded, y_train_loaded, _, _ = bench_loaded._generate_data()

        assert_array_almost_equal(X_train_original, X_train_loaded)
        assert_array_almost_equal(y_train_original, y_train_loaded)

        # Check that metadata is preserved
        assert bench_loaded.metadata.name == bench.metadata.name
        assert bench_loaded.metadata.train_size == bench.metadata.train_size

    def test_pickle_with_custom_params(self):
        """Test pickling with custom parameters."""
        bench = KozaQuarticBenchmark(
            random_state=123,
            train_samples=50,
            test_samples=200,
            x_range=(-2, 2),
            train_type="grid",
            test_type="uniform",
        )

        # Pickle and unpickle
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(bench, f)
            f.flush()
            f.seek(0)
            bench_loaded = pickle.load(f)

        # Check parameters are preserved
        assert bench_loaded.random_state == bench.random_state
        assert bench_loaded.train_samples == bench.train_samples
        assert bench_loaded.test_samples == bench.test_samples
        assert bench_loaded.x_range == bench.x_range
        assert bench_loaded.train_type == bench.train_type
        assert bench_loaded.test_type == bench.test_type


class TestKozaQuarticReproducibility:
    """Test reproducibility across runs."""

    def test_reproducibility(self):
        """Test that same random state produces same data ACROSS INSTANCES."""
        bench1 = KozaQuarticBenchmark(random_state=123)
        bench2 = KozaQuarticBenchmark(random_state=123)

        # First call - should match across instances
        X1_1, y1_1, X1_test_1, y1_test_1 = bench1._generate_data()
        X2_1, y2_1, X2_test_1, y2_test_1 = bench2._generate_data()

        assert_array_almost_equal(X1_1, X2_1)
        assert_array_almost_equal(y1_1, y2_1)
        assert_array_almost_equal(X1_test_1, X2_test_1)
        assert_array_almost_equal(y1_test_1, y2_test_1)

        # Second call - should also match across instances
        X1_2, y1_2, X1_test_2, y1_test_2 = bench1._generate_data()
        X2_2, y2_2, X2_test_2, y2_test_2 = bench2._generate_data()

        assert_array_almost_equal(X1_2, X2_2)
        assert_array_almost_equal(y1_2, y2_2)
        assert_array_almost_equal(X1_test_2, X2_test_2)
        assert_array_almost_equal(y1_test_2, y2_test_2)

        # BUT: First call and second call for same instance should be DIFFERENT
        assert not np.array_equal(X1_1, X1_2)
        assert not np.array_equal(y1_1, y1_2)

    def test_hash_consistency(self):
        """Test that data hashes are consistent (for regression testing)."""
        # Known good hashes for Koza Quartic with seed 42
        bench = KozaQuarticBenchmark(random_state=42)
        X_train, y_train, X_test, y_test = bench._generate_data()

        # Create hashes of the data
        train_hash = hashlib.md5(X_train.tobytes()).hexdigest()
        test_hash = hashlib.md5(X_test.tobytes()).hexdigest()

        # These hashes should remain constant across versions
        expected_train_hash = "04e69cc9af3415de4ab2820aa3e5ea99"
        expected_test_hash = "6e5a2e389c3e1f67cf1cbcac6037340c"

        assert train_hash == expected_train_hash
        assert test_hash == expected_test_hash


class TestKozaQuarticEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test with single sample."""
        bench = KozaQuarticBenchmark(train_samples=1, test_samples=1)
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert X_train.shape == (1, 1)
        assert y_train.shape == (1,)
        assert X_test.shape == (1, 1)
        assert y_test.shape == (1,)

    def test_large_sample_size(self):
        """Test with large sample sizes."""
        bench = KozaQuarticBenchmark(train_samples=10000, test_samples=10000)
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert len(X_train) == 10000
        assert len(y_train) == 10000
        assert len(X_test) == 10000
        assert len(y_test) == 10000

    def test_extreme_ranges(self):
        """Test with extreme input ranges."""
        bench = KozaQuarticBenchmark(x_range=(-1e6, 1e6))
        X_train, y_train, _, _ = bench._generate_data()

        assert np.all((X_train >= -1e6) & (X_train <= 1e6))

        # Function should still compute (might be huge numbers)
        assert np.all(np.isfinite(y_train))

    def test_summary_method(self):
        """Test summary method if implemented."""
        bench = KozaQuarticBenchmark(random_state=42, train_samples=75)

        # Check if summary method exists (optional)
        if hasattr(bench, "summary"):
            summary = bench.summary()
            assert summary["name"] == "Koza Quartic"
            assert summary["train_samples"] == 75
            assert summary["random_state"] == 42


class TestKozaQuarticScientificValidity:
    """Tests to ensure benchmark is scientifically sound."""

    def test_function_continuity(self):
        """Test that function is continuous in its domain."""
        bench = KozaQuarticBenchmark()

        # Generate dense sampling
        X = np.linspace(-1, 1, 1000).reshape(-1, 1)
        y = bench._quartic_function(X)

        # Check for discontinuities (large jumps)
        diffs = np.abs(np.diff(y))

        # For a polynomial, differences should be smooth
        # Maximum difference should be at boundaries where slope is highest
        max_diff = np.max(diffs)
        assert max_diff < 10.0  # Reasonable threshold

        # Check that differences change smoothly (second differences)
        second_diffs = np.abs(np.diff(diffs))
        assert np.all(second_diffs < 1.0)

    def test_function_monotonicity(self):
        """Test function behavior in different regions."""
        bench = KozaQuarticBenchmark()

        # Function should be decreasing for x < -0.5, increasing for x > 0
        X_decreasing = np.linspace(-1, -0.6, 10).reshape(-1, 1)
        y_decreasing = bench._quartic_function(X_decreasing)

        X_increasing = np.linspace(0, 1, 10).reshape(-1, 1)
        y_increasing = bench._quartic_function(X_increasing)

        # Check monotonicity
        assert np.all(np.diff(y_decreasing) < 0)  # Strictly decreasing
        assert np.all(np.diff(y_increasing) > 0)  # Strictly increasing

    def test_training_coverage(self):
        """Test that training data provides good coverage of the domain."""
        bench = KozaQuarticBenchmark(train_samples=100, train_type="uniform")
        X_train, _, _, _ = bench._generate_data()

        # Flatten if needed
        if X_train.ndim > 1:
            X_train = X_train.flatten()

        low, high = bench.x_range
        range_width = high - low

        # More realistic expectations for 100 samples
        percentiles = np.percentile(X_train, [10, 25, 50, 75, 90])

        # 10th percentile should be in the bottom 30% of the range
        assert (
            percentiles[0] < low + 0.3 * range_width
        ), f"10th percentile {percentiles[0]:.3f} too high"

        # 90th percentile should be in the top 30% of the range
        assert (
            percentiles[-1] > high - 0.3 * range_width
        ), f"90th percentile {percentiles[-1]:.3f} too low"

        # Check that min and max are near the boundaries
        assert (
            X_train.min() < low + 0.1 * range_width
        ), f"Min {X_train.min():.3f} not near lower bound"
        assert (
            X_train.max() > high - 0.1 * range_width
        ), f"Max {X_train.max():.3f} not near upper bound"

    def test_koza_quartic_reproducibility(self):
        """Test that Koza Quartic generates same data with same seed."""
        # First instance
        b1 = KozaQuarticBenchmark(random_state=42)
        X1_train, y1_train, X1_test, y1_test = b1.load_data()

        # Second instance with same seed
        b2 = KozaQuarticBenchmark(random_state=42)
        X2_train, y2_train, X2_test, y2_test = b2.load_data()

        # Should be identical
        np.testing.assert_array_equal(X1_train, X2_train)
        np.testing.assert_array_equal(y1_train, y2_train)
        np.testing.assert_array_equal(X1_test, X2_test)
        np.testing.assert_array_equal(y1_test, y2_test)

        # Different seed :: different data
        b3 = KozaQuarticBenchmark(random_state=99)
        X3_train, y3_train, _, _ = b3.load_data()

        # Should NOT be identical
        assert not np.array_equal(X1_train, X3_train)
