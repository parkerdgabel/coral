"""Tests for JSON utilities that handle numpy types."""

import io
import json

import numpy as np
import pytest

from coral.utils.json_utils import NumpyJSONEncoder, dump_numpy, dumps_numpy


class TestNumpyJSONEncoder:
    """Tests for NumpyJSONEncoder class."""

    def test_encode_numpy_integer(self):
        """Test encoding numpy integers."""
        data = {"value": np.int32(42)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["value"] == 42
        assert isinstance(parsed["value"], int)

    def test_encode_numpy_int64(self):
        """Test encoding numpy int64."""
        data = {"value": np.int64(123456789)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["value"] == 123456789

    def test_encode_numpy_float(self):
        """Test encoding numpy floats."""
        data = {"value": np.float32(3.14)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert abs(parsed["value"] - 3.14) < 0.01
        assert isinstance(parsed["value"], float)

    def test_encode_numpy_float64(self):
        """Test encoding numpy float64."""
        data = {"value": np.float64(2.718281828)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert abs(parsed["value"] - 2.718281828) < 0.0001

    def test_encode_numpy_array(self):
        """Test encoding numpy arrays."""
        data = {"array": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["array"] == [1, 2, 3]
        assert isinstance(parsed["array"], list)

    def test_encode_numpy_2d_array(self):
        """Test encoding 2D numpy arrays."""
        data = {"matrix": np.array([[1, 2], [3, 4]])}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["matrix"] == [[1, 2], [3, 4]]

    def test_encode_numpy_dtype(self):
        """Test encoding numpy dtypes."""
        data = {"dtype": np.dtype("float32")}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["dtype"] == "float32"
        assert isinstance(parsed["dtype"], str)

    def test_encode_numpy_bool(self):
        """Test encoding numpy booleans."""
        data = {"flag_true": np.bool_(True), "flag_false": np.bool_(False)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["flag_true"] is True
        assert parsed["flag_false"] is False
        assert isinstance(parsed["flag_true"], bool)

    def test_encode_numpy_void(self):
        """Test encoding numpy void types."""
        # Create a structured array with void dtype
        dt = np.dtype([("x", np.int32), ("y", np.float32)])
        arr = np.array([(1, 2.0)], dtype=dt)
        void_val = arr[0]

        data = {"void": void_val}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        # void is serialized as string
        assert isinstance(parsed["void"], str)

    def test_encode_numpy_complex(self):
        """Test encoding numpy complex numbers."""
        data = {"complex": np.complex64(1 + 2j)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        # Complex is serialized as string
        assert isinstance(parsed["complex"], str)
        assert "1" in parsed["complex"]
        assert "2" in parsed["complex"]

    def test_encode_regular_types_unchanged(self):
        """Test that regular Python types are encoded normally."""
        data = {
            "int": 42,
            "float": 3.14,
            "string": "hello",
            "list": [1, 2, 3],
            "dict": {"a": 1},
            "bool": True,
            "none": None,
        }
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed == data

    def test_encode_mixed_types(self):
        """Test encoding mixed numpy and Python types."""
        data = {
            "np_int": np.int32(10),
            "py_int": 20,
            "np_float": np.float64(1.5),
            "py_float": 2.5,
            "np_array": np.array([1, 2]),
            "py_list": [3, 4],
        }
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["np_int"] == 10
        assert parsed["py_int"] == 20
        assert abs(parsed["np_float"] - 1.5) < 0.001
        assert parsed["py_float"] == 2.5
        assert parsed["np_array"] == [1, 2]
        assert parsed["py_list"] == [3, 4]

    def test_encode_nested_numpy_types(self):
        """Test encoding nested structures with numpy types."""
        data = {
            "outer": {
                "inner": {
                    "value": np.int64(100),
                    "array": np.array([1.0, 2.0, 3.0]),
                }
            }
        }
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        assert parsed["outer"]["inner"]["value"] == 100
        assert parsed["outer"]["inner"]["array"] == [1.0, 2.0, 3.0]

    def test_encode_unsupported_type_raises(self):
        """Test that unsupported types raise TypeError."""

        class CustomClass:
            pass

        data = {"custom": CustomClass()}

        with pytest.raises(TypeError):
            json.dumps(data, cls=NumpyJSONEncoder)


class TestDumpsNumpy:
    """Tests for dumps_numpy function."""

    def test_dumps_simple(self):
        """Test simple serialization."""
        data = {"value": np.int32(42)}
        result = dumps_numpy(data)

        assert '"value": 42' in result

    def test_dumps_with_indent(self):
        """Test serialization with indent."""
        data = {"a": 1, "b": 2}
        result = dumps_numpy(data, indent=2)

        assert "\n" in result
        assert "  " in result

    def test_dumps_with_sort_keys(self):
        """Test serialization with sorted keys."""
        data = {"c": 3, "a": 1, "b": 2}
        result = dumps_numpy(data, sort_keys=True)

        # Keys should appear in alphabetical order
        a_pos = result.find('"a"')
        b_pos = result.find('"b"')
        c_pos = result.find('"c"')

        assert a_pos < b_pos < c_pos

    def test_dumps_numpy_array(self):
        """Test serialization of numpy array."""
        data = {"weights": np.random.randn(3, 3).astype(np.float32)}
        result = dumps_numpy(data)

        parsed = json.loads(result)
        assert len(parsed["weights"]) == 3
        assert len(parsed["weights"][0]) == 3


class TestDumpNumpy:
    """Tests for dump_numpy function."""

    def test_dump_to_file(self):
        """Test dumping to file-like object."""
        data = {"value": np.int64(12345)}
        fp = io.StringIO()

        dump_numpy(data, fp)
        fp.seek(0)
        result = fp.read()

        parsed = json.loads(result)
        assert parsed["value"] == 12345

    def test_dump_with_indent(self):
        """Test dumping with indent."""
        data = {"a": 1, "b": np.float32(2.5)}
        fp = io.StringIO()

        dump_numpy(data, fp, indent=4)
        fp.seek(0)
        result = fp.read()

        assert "\n" in result
        assert "    " in result

    def test_dump_numpy_types(self):
        """Test dumping various numpy types."""
        data = {
            "int8": np.int8(1),
            "int16": np.int16(2),
            "int32": np.int32(3),
            "int64": np.int64(4),
            "float16": np.float16(1.5),
            "float32": np.float32(2.5),
            "float64": np.float64(3.5),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3]),
        }
        fp = io.StringIO()

        dump_numpy(data, fp)
        fp.seek(0)
        result = fp.read()

        parsed = json.loads(result)
        assert parsed["int8"] == 1
        assert parsed["int16"] == 2
        assert parsed["int32"] == 3
        assert parsed["int64"] == 4
        assert parsed["bool"] is True
        assert parsed["array"] == [1, 2, 3]
