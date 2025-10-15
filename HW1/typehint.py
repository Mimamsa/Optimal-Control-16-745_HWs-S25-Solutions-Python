import jax
import numpy as np
from typing import TypeVar, Generic

Shape = TypeVar("Shape")
DType = TypeVar("DType")

class NDArray(np.ndarray, Generic[Shape, DType]):
    """  
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass


class JNDArray(jax.Array, Generic[Shape, DType]):
    """  
    Use this to type-annotate jax.numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass

