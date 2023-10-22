import tempfile
import os
import time
import torch as t
from torch import nn
import transformers
import joblib
import requests
import logging
import http
from functools import wraps
from typing import Optional, Iterator, cast, TypeVar, Generic, Callable

mem = joblib.Memory(tempfile.gettempdir() + "/joblib_cache")
DEBUG_TOLERANCES = os.getenv("DEBUG_TOLERANCES")


def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    """Assert that actual and expected are exactly equal (to floating point precision)."""
    mask = actual == expected
    if not mask.all().item():
        bad = mask.nonzero()
        msg = f"Did not match at {len(bad)} indexes: {bad[:10]}{'...' if len(bad) > 10 else ''}"
        raise AssertionError(f"{msg}\nActual:\n{actual}\nExpected:\n{expected}")


def test_is_equal(actual: t.Tensor, expected: t.Tensor, test_name: str) -> None:
    try:
        run_and_report(assert_all_equal, test_name, actual, expected)
    except AssertionError as e:
        print(f"Test failed: {test_name}")
        raise e


def assert_shape_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"expected shape={expected.shape}, got {actual.shape}")


def allclose(actual: t.Tensor, expected: t.Tensor, rtol=1e-4) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    right = rtol * expected.abs()
    num_wrong = (left > right).sum().item()
    if num_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(
            f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance"
        )
    elif DEBUG_TOLERANCES:
        print(f"Test passed with max absolute deviation of {left.max()}")


def allclose_atol(actual: t.Tensor, expected: t.Tensor, atol: float) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    num_wrong = (left > atol).sum().item()
    if num_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(
            f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance"
        )
    elif DEBUG_TOLERANCES:
        print(f"Test passed with max absolute deviation of {left.max()}")


def allclose_scalar(actual: float, expected: float, rtol=1e-4) -> None:
    left = abs(actual - expected)
    right = rtol * abs(expected)
    wrong = left > right
    if wrong:
        raise AssertionError(
            f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}"
        )
    elif DEBUG_TOLERANCES:
        print(f"Test passed with absolute deviation of {left}")


def allclose_scalar_atol(actual: float, expected: float, atol: float) -> None:
    left = abs(actual - expected)
    wrong = left > atol
    if wrong:
        raise AssertionError(
            f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}"
        )
    elif DEBUG_TOLERANCES:
        print(f"Test passed with absolute deviation of {left}")
