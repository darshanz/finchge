import pickle
import tempfile

import numpy as np
from numpy.testing import assert_array_equal

from finchge.benchmarks.logic.multiplexer import (
    Multiplexer6Benchmark,
    Multiplexer11Benchmark,
)
from finchge.grammar.grammar import Grammar


class TestMultiplexer6Benchmark:
    def test_metadata(self):
        bench = Multiplexer6Benchmark()

        assert bench.metadata.name == "6-bit Multiplexer"
        assert bench.metadata.category == "logic"
        assert bench.metadata.input_dim == 6
        assert bench.metadata.output_dim == 1
        assert bench.metadata.train_size == 64
        assert bench.metadata.test_size == 64
        assert "Koza" in bench.metadata.reference

    def test_data_shapes(self):
        bench = Multiplexer6Benchmark()
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert X_train.shape == (64, 6)
        assert y_train.shape == (64,)
        assert X_test.shape == (64, 6)
        assert y_test.shape == (64,)

        # Values should be 0 or 1
        assert np.all((X_train == 0) | (X_train == 1))
        assert np.all((y_train == 0) | (y_train == 1))

        assert np.all(X_train == X_train.astype(int))
        assert np.all(y_train == y_train.astype(int))

    def test_all_combinations_present(self):
        bench = Multiplexer6Benchmark()
        X_train, _, _, _ = bench._generate_data()

        combinations = set(tuple(row) for row in X_train)
        assert len(combinations) == 64

        # Verify specific patterns are present
        assert (0, 0, 0, 0, 0, 0) in combinations
        assert (1, 1, 1, 1, 1, 1) in combinations
        assert (0, 0, 0, 1, 0, 1) in combinations

    def test_multiplexer_logic_6bit(self):
        bench = Multiplexer6Benchmark()
        X, y, _, _ = bench._generate_data()

        for i in range(64):
            bits = X[i]

            # 6-bit multiplexer: 2 address bits (indices 0,1), 4 data bits (indices 2-5)
            # Address calculation (LSB first as in implementation)
            address = bits[0] | (
                bits[1] << 1
            )  # LSB first: bit0 is LSB, bit1 is MSB of address

            # Data bits start at index 2
            expected = bits[2 + address]

            assert (
                y[i] == expected
            ), f"Row {i}: bits={bits}, address={address}, expected={expected}, got={y[i]}"

    def test_specific_cases_6bit(self):
        bench = Multiplexer6Benchmark()
        X, y, _, _ = bench._generate_data()

        def find_row(bits):
            for i, row in enumerate(X):
                if np.array_equal(row, bits):
                    return i
            return None

        # Format: [A0, A1, D0, D1, D2, D3] where A0 is LSB of address

        # Test case: address=00 (0), data=0000 ---> select D0 (0)
        row = find_row([0, 0, 0, 0, 0, 0])
        assert row is not None, "Row [0,0,0,0,0,0] not found"
        assert y[row] == 0, f"Expected 0, got {y[row]}"

        # Test case: address =00 (0), data=1000 ----> select D0 (1)
        row = find_row([0, 0, 1, 0, 0, 0])
        assert row is not None
        assert y[row] == 1, f"Expected 1, got {y[row]}"

        # Test case: address=01 (1), data=0100 ---> select D1 (1)
        row = find_row([1, 0, 0, 1, 0, 0])  # A0=1, A1=0 (address 01), D1=1
        assert row is not None
        assert y[row] == 1, f"Expected 1, got {y[row]}"

        # Test case: address=10 (2), data=0010 ----> select D2 (1)
        row = find_row([0, 1, 0, 0, 1, 0])  # A0=0, A1=1 (address 10), D2=1
        assert row is not None
        assert y[row] == 1, f"Expected 1, got {y[row]}"

        # Test case: address=11 (3), data=0001 ----> select D3 (1)
        row = find_row([1, 1, 0, 0, 0, 1])  # A0=1, A1=1 (address 11), D3=1
        assert row is not None
        assert y[row] == 1, f"Expected 1, got {y[row]}"

        # Test case: all ones
        row = find_row([1, 1, 1, 1, 1, 1])
        assert row is not None
        assert y[row] == 1  # address 11 selects D3 which is 1

    def test_deterministic(self):
        bench = Multiplexer6Benchmark()

        X1, y1, X2, y2 = bench._generate_data()
        X3, y3, X4, y4 = bench._generate_data()

        assert_array_equal(X1, X3)
        assert_array_equal(y1, y3)
        assert_array_equal(X2, X4)
        assert_array_equal(y2, y4)

    def test_train_test_identical(self):
        bench = Multiplexer6Benchmark()
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert_array_equal(X_train, X_test)
        assert_array_equal(y_train, y_test)

    def test_grammar_content(self):
        bench = Multiplexer6Benchmark()
        grammar = bench.grammar()

        assert "if" in grammar.terminals
        assert "and" in grammar.terminals
        assert "or" in grammar.terminals
        assert "not" in grammar.terminals

        for i in range(6):
            assert f"x{i}" in grammar.terminals

        # Should not have extra variables
        assert "x6" not in grammar.terminals

    def test_pickle_roundtrip(self):
        bench = Multiplexer6Benchmark()
        X_original, y_original, _, _ = bench._generate_data()

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(bench, f)
            f.flush()
            f.seek(0)
            bench_loaded = pickle.load(f)

        X_loaded, y_loaded, _, _ = bench_loaded._generate_data()

        assert_array_equal(X_original, X_loaded)
        assert_array_equal(y_original, y_loaded)
        assert bench_loaded.metadata.name == bench.metadata.name

    def test_repr(self):
        bench = Multiplexer6Benchmark()
        repr_str = repr(bench)
        assert "Multiplexer6Benchmark" in repr_str
        assert "64" in repr_str


class TestMultiplexer11Benchmark:
    def test_metadata(self):
        bench = Multiplexer11Benchmark()

        assert bench.metadata.name == "11-bit Multiplexer"
        assert bench.metadata.category == "logic"
        assert bench.metadata.input_dim == 11
        assert bench.metadata.output_dim == 1
        assert bench.metadata.train_size == 2048
        assert bench.metadata.test_size == 2048

    def test_data_shapes(self):
        bench = Multiplexer11Benchmark()
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert X_train.shape == (2048, 11)
        assert y_train.shape == (2048,)
        assert X_test.shape == (2048, 11)
        assert y_test.shape == (2048,)

    def test_all_combinations_present(self):
        bench = Multiplexer11Benchmark()
        X_train, _, _, _ = bench._generate_data()

        combinations = set(tuple(row) for row in X_train)
        assert len(combinations) == 2048

    def test_multiplexer_logic_11bit(self):
        bench = Multiplexer11Benchmark()
        X, y, _, _ = bench._generate_data()

        for i in range(100):
            bits = X[i]

            # 11-bit multiplexer: 3 address bits (indices 0,1,2), 8 data bits (indices 3-10)
            # Address calculation (LSB first)
            address = bits[0] | (bits[1] << 1) | (bits[2] << 2)

            # Data bits start at index 3
            expected = bits[3 + address]

            assert (
                y[i] == expected
            ), f"Row {i}: bits={bits}, address={address}, expected={expected}, got={y[i]}"

    def test_specific_cases_11bit(self):
        bench = Multiplexer11Benchmark()
        X, y, _, _ = bench._generate_data()

        def find_row(bits):
            for i, row in enumerate(X):
                if np.array_equal(row, bits):
                    return i
            return None

        # Format: [A0, A1, A2, D0, D1, D2, D3, D4, D5, D6, D7] where A0 is LSB

        # Test case: address=000 (0), data=00000001 -> select D0 (1)
        row = find_row([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        assert row is not None
        assert y[row] == 1

        # Test case: address=001 (1), data=00000010 -> select D1 (1)
        row = find_row([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert row is not None
        assert y[row] == 1

        # Test case: address=010 (2), data=00000100 -> select D2 (1)
        row = find_row([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert row is not None
        assert y[row] == 1

        # Test case: address=011 (3), data=00001000 -> select D3 (1)
        row = find_row([1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        assert row is not None
        assert y[row] == 1

        # Test case: address=100 (4), data=00010000 -> select D4 (1)
        row = find_row([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        assert row is not None
        assert y[row] == 1

        # Test case: address=101 (5), data=00100000 -> select D5 (1)
        row = find_row([1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
        assert row is not None
        assert y[row] == 1

        # Test case: address=110 (6), data=01000000 -> select D6 (1)
        row = find_row([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
        assert row is not None
        assert y[row] == 1

        # Test case: address=111 (7), data=10000000 -> select D7 (1)
        row = find_row([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        assert row is not None
        assert y[row] == 1

    def test_deterministic(self):
        bench = Multiplexer11Benchmark()

        X1, y1, X2, y2 = bench._generate_data()
        X3, y3, X4, y4 = bench._generate_data()

        assert_array_equal(X1, X3)
        assert_array_equal(y1, y3)
        assert_array_equal(X2, X4)
        assert_array_equal(y2, y4)

    def test_train_test_identical(self):
        bench = Multiplexer11Benchmark()
        X_train, y_train, X_test, y_test = bench._generate_data()

        assert_array_equal(X_train, X_test)
        assert_array_equal(y_train, y_test)

    def test_grammar_content(self):
        bench = Multiplexer11Benchmark()
        grammar = Grammar(bench.grammar_str())

        # Should have all needed functions
        assert "if" in grammar.terminals
        assert "and" in grammar.terminals
        assert "or" in grammar.terminals
        assert "not" in grammar.terminals

        # Should have all 11 variables
        for i in range(11):
            assert f"x{i}" in grammar.terminals

        # Should not have extra variables
        assert "x11" not in grammar.terminals

    def test_pickle_roundtrip(self):
        bench = Multiplexer11Benchmark()
        X_original, y_original, _, _ = bench._generate_data()

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(bench, f)
            f.flush()
            f.seek(0)
            bench_loaded = pickle.load(f)

        X_loaded, y_loaded, _, _ = bench_loaded._generate_data()

        assert_array_equal(X_original, X_loaded)
        assert_array_equal(y_original, y_loaded)

    def test_repr(self):
        bench = Multiplexer11Benchmark()
        repr_str = repr(bench)
        assert "Multiplexer11Benchmark" in repr_str
        assert "2048" in repr_str


class TestMultiplexerCommon:
    def test_no_random_state_effect(self):
        bench1 = Multiplexer6Benchmark(random_state=42)
        bench2 = Multiplexer6Benchmark(random_state=123)

        X1, y1, _, _ = bench1._generate_data()
        X2, y2, _, _ = bench2._generate_data()

        assert_array_equal(X1, X2)
        assert_array_equal(y1, y2)

        # Same for 11-bit
        bench1 = Multiplexer11Benchmark(random_state=42)
        bench2 = Multiplexer11Benchmark(random_state=123)

        X1, y1, _, _ = bench1._generate_data()
        X2, y2, _, _ = bench2._generate_data()

        assert_array_equal(X1, X2)
        assert_array_equal(y1, y2)

    def test_boolean_values(self):
        bench6 = Multiplexer6Benchmark()
        X6, y6, _, _ = bench6._generate_data()

        assert np.all((X6 == 0) | (X6 == 1))
        assert np.all((y6 == 0) | (y6 == 1))

        bench11 = Multiplexer11Benchmark()
        X11, y11, _, _ = bench11._generate_data()

        assert np.all((X11 == 0) | (X11 == 1))
        assert np.all((y11 == 0) | (y11 == 1))

    def test_address_data_separation(self):
        bench6 = Multiplexer6Benchmark()
        X6, _, _, _ = bench6._generate_data()

        # First 3 bits are address, last 3 are data
        assert np.all(X6[:, :3].max() <= 1)
        assert np.all(X6[:, 3:].max() <= 1)

        bench11 = Multiplexer11Benchmark()
        X11, _, _, _ = bench11._generate_data()

        # First 4 bits are address, last 7 are data
        assert np.all(X11[:, :4].max() <= 1)
        assert np.all(X11[:, 4:].max() <= 1)
