# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper around the OpenDS async C API (libopends_*.so).

Drop-in replacement for :mod:`_cufile_async` that uses OpenDS instead of
cuFile. The API surface (AsyncHandle, Submission, register_*/deregister_*)
is identical so that :mod:`gds_context` can use either backend
transparently.

OpenDS is structurally compatible with cuFile and keeps cuFile's file
semantics: the slab is a regular file on a mounted filesystem, opened with
O_DIRECT and registered by fd. The aisio backend resolves the file's
extents and DMAs straight from the NVMe LBAs into GPU memory, so the same
``lmcache_gds_slab.bin`` on the same filesystem can be driven by either
cuFile or OpenDS. That keeps a cuFile-vs-OpenDS comparison free of the
raw-device-vs-filesystem confound.

OpenDS ships one shared object per backend (``libopends_aisio.so``,
``libopends_gds.so``, ``libopends_ref.so``), each exporting the same
``opends_*`` symbols. ``LMCACHE_OPENDS_LIB`` selects which one to dlopen;
the default is the aisio backend.
"""

# Standard
import ctypes
import ctypes.util
import os
from typing import Any, Optional

_DEFAULT_LIB = "libopends_aisio.so"

# --- libopends lazy loading ------------------------------------------

_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("LMCACHE_OPENDS_LIB", "") or _DEFAULT_LIB
    _lib = ctypes.CDLL(path)
    _declare_signatures(_lib)
    return _lib


# --- OpenDS C types --------------------------------------------------


class _opends_error_t(ctypes.Structure):
    _fields_ = [
        ("err", ctypes.c_int),
        ("dev_err", ctypes.c_int),
    ]


def _declare_signatures(lib: ctypes.CDLL) -> None:
    lib.opends_driver_open.argtypes = []
    lib.opends_driver_open.restype = _opends_error_t

    lib.opends_driver_close.argtypes = []
    lib.opends_driver_close.restype = _opends_error_t

    lib.opends_handle_register.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # opends_handle_t *fh
        ctypes.c_int,                     # int fd
    ]
    lib.opends_handle_register.restype = _opends_error_t

    lib.opends_handle_deregister.argtypes = [ctypes.c_void_p]
    lib.opends_handle_deregister.restype = None

    lib.opends_buf_register.argtypes = [
        ctypes.c_void_p,   # const void *buf_base
        ctypes.c_size_t,   # size_t size
        ctypes.c_int,      # int flags
    ]
    lib.opends_buf_register.restype = _opends_error_t

    lib.opends_buf_deregister.argtypes = [ctypes.c_void_p]
    lib.opends_buf_deregister.restype = _opends_error_t

    lib.opends_read_async.argtypes = [
        ctypes.c_void_p,                   # opends_handle_t fh
        ctypes.c_void_p,                   # void *buf_base
        ctypes.POINTER(ctypes.c_size_t),   # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),    # off_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),    # off_t *buf_offset_p
        ctypes.POINTER(ctypes.c_int64),    # ssize_t *bytes_read_p
        ctypes.c_void_p,                   # opends_stream_t stream
    ]
    lib.opends_read_async.restype = _opends_error_t

    lib.opends_write_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    lib.opends_write_async.restype = _opends_error_t

    lib.opends_stream_register.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.opends_stream_register.restype = _opends_error_t

    lib.opends_stream_deregister.argtypes = [ctypes.c_void_p]
    lib.opends_stream_deregister.restype = _opends_error_t

    lib.opends_op_status_error.argtypes = [ctypes.c_int]
    lib.opends_op_status_error.restype = ctypes.c_char_p


# The aisio backend ignores the stream-registration flags; cuFile's
# FIXED_* bits have no OpenDS equivalent.
_STREAM_REGISTER_FLAGS = 0


# --- Error checking --------------------------------------------------


def _check(err: _opends_error_t, op: str) -> None:
    if err.err != 0:
        lib = _get_lib()
        detail = lib.opends_op_status_error(err.err)
        text = detail.decode() if detail else "unknown"
        raise RuntimeError(
            f"{op} failed: opendsError(err={err.err} [{text}], "
            f"dev_err={err.dev_err})"
        )


# --- Driver lifecycle ------------------------------------------------

_driver_opened = False


def _ensure_driver_open() -> None:
    global _driver_opened
    if _driver_opened:
        return
    lib = _get_lib()
    _check(lib.opends_driver_open(), "opends_driver_open")
    _driver_opened = True


def close_driver() -> None:
    global _driver_opened
    if not _driver_opened:
        return
    lib = _get_lib()
    try:
        _check(lib.opends_driver_close(), "opends_driver_close")
    finally:
        _driver_opened = False


# --- Handle registration ---------------------------------------------


def register_handle(fd: int) -> int:
    """Register an open slab fd and return the raw opends_handle_t.

    Mirrors _cufile_async.register_handle(fd): the caller owns the fd
    (an O_DIRECT open of the slab file) and closes it on registration
    failure; wrap the pair via AsyncHandle.from_fd.
    """
    _ensure_driver_open()
    lib = _get_lib()
    handle = ctypes.c_void_p()
    _check(
        lib.opends_handle_register(ctypes.byref(handle), ctypes.c_int(fd)),
        "opends_handle_register",
    )
    if handle.value is None:
        raise RuntimeError("opends_handle_register returned a null handle")
    return handle.value


def deregister_handle(handle: int) -> None:
    """Reverse of register_handle (opends_handle_deregister)."""
    lib = _get_lib()
    lib.opends_handle_deregister(ctypes.c_void_p(handle))


# --- Buffer / stream registration ------------------------------------


def register_buffer(buf: "torch.Tensor") -> None:  # noqa: F821
    # Third Party
    import torch  # noqa: F401

    if not buf.is_cuda:
        raise ValueError("register_buffer: tensor must be on CUDA")
    _ensure_driver_open()
    lib = _get_lib()
    nbytes = buf.numel() * buf.element_size()
    _check(
        lib.opends_buf_register(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(0),
        ),
        "opends_buf_register",
    )


def deregister_buffer(buf: "torch.Tensor") -> None:  # noqa: F821
    lib = _get_lib()
    _check(
        lib.opends_buf_deregister(ctypes.c_void_p(buf.data_ptr())),
        "opends_buf_deregister",
    )


def register_stream(raw_stream: int) -> None:
    _ensure_driver_open()
    lib = _get_lib()
    _check(
        lib.opends_stream_register(
            ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
        ),
        "opends_stream_register",
    )


def deregister_stream(raw_stream: int) -> None:
    lib = _get_lib()
    _check(
        lib.opends_stream_deregister(ctypes.c_void_p(raw_stream)),
        "opends_stream_deregister",
    )


# --- Submission + AsyncHandle ----------------------------------------


class Submission:
    """One in-flight opends_read_async / opends_write_async.

    Mirrors _cufile_async.Submission: holds ctypes storage that must
    stay alive until the stream executes the op.
    """

    __slots__ = ("_size", "_file_offset", "_buf_offset", "_bytes_done")

    def __init__(self, size: int, file_offset: int, buf_offset: int) -> None:
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_int64(file_offset)
        self._buf_offset = ctypes.c_int64(buf_offset)
        self._bytes_done = ctypes.c_int64(0)

    @property
    def bytes_done(self) -> int:
        return self._bytes_done.value


class AsyncHandle:
    """Slab file + OpenDS handle wrapper, API-compatible with _cufile_async."""

    __slots__ = ("_fd", "_handle", "path", "writable")

    def __init__(
        self,
        path: str,
        writable: bool = True,
        fallocate_size: Optional[int] = None,
        mode: int = 0o644,
    ) -> None:
        flags = os.O_DIRECT
        if writable:
            flags |= os.O_CREAT | os.O_RDWR
        else:
            flags |= os.O_RDONLY
        fd = os.open(path, flags, mode)
        try:
            if fallocate_size is not None and writable:
                os.posix_fallocate(fd, 0, fallocate_size)
            handle = register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._handle = handle
        self.path = path
        self.writable = writable

    @classmethod
    def from_fd(
        cls,
        fd: int,
        handle: int,
        path: str,
        writable: bool = False,
    ) -> "AsyncHandle":
        """Wrap an already-opened fd and registered OpenDS handle.

        Matches _cufile_async.AsyncHandle.from_fd.
        """
        obj = cls.__new__(cls)
        obj._fd = fd
        obj._handle = handle
        obj.path = path
        obj.writable = writable
        return obj

    @property
    def fd(self) -> int:
        return self._fd

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        lib = _get_lib()
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            lib.opends_read_async(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "opends_read_async",
        )
        return sub

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        lib = _get_lib()
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            lib.opends_write_async(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "opends_write_async",
        )
        return sub

    def close(self) -> None:
        if self._fd < 0:
            return
        try:
            deregister_handle(self._handle)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1

    def __enter__(self) -> "AsyncHandle":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
