# SPDX-License-Identifier: Apache-2.0
"""OpenDS implementation of the shared GDS contracts.

OpenDS is a vendor-neutral accelerator direct storage API modeled on cuFile.
Each OpenDS backend is its own shared object exporting the same ``opends_*``
symbols: ``libopends_aisio.so`` (NVMe-to-GPU PCIe P2P DMA through xNVMe, with
no kernel driver in the data path), ``libopends_cufile.so`` (NVIDIA cuFile) and
``libopends_ref.so`` (POSIX reference). As for the OpenDS Python package,
``OPENDS_BACKEND`` names the backend (default ``aisio``) and ``OPENDS_LIBRARY``
overrides the library path.

Every OpenDS backend registers only ``O_DIRECT`` descriptors, so the slab is
always opened with direct IO. ``opends_stream_read``/``opends_stream_write``
take cuFile's ReadAsync/WriteAsync pointer arguments, so each Submission keeps
its native storage until the stream runs it.
"""

# Standard
from typing import Optional, cast
import ctypes
import ctypes.util
import os
import threading

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSHandle, FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import Submission

_DEFAULT_OPENDS_BACKEND = "aisio"

_OPENDS_SUCCESS = 0

# cuFile's FIXED_BUF_OFFSET | FIXED_FILE_OFFSET | FIXED_FILE_SIZE. The cufile
# backend forwards the flags to cuFileStreamRegister; aisio and ref ignore them.
_STREAM_REGISTER_FLAGS = 0x7


class _OpendsError(ctypes.Structure):
    """ctypes mirror of ``opends_error_t`` (opends.h).

    ``err`` is the ``opends_op_error_t`` status (0 == success) and ``dev_err``
    the backend-specific subcode, such as a CUDA error from aisio.
    """

    _fields_ = [("err", ctypes.c_int), ("dev_err", ctypes.c_int)]


def _library_path() -> str:
    """Resolve the OpenDS shared object from the OpenDS environment variables."""
    explicit = os.environ.get("OPENDS_LIBRARY")
    if explicit:
        return explicit
    name = f"opends_{os.environ.get('OPENDS_BACKEND', _DEFAULT_OPENDS_BACKEND)}"
    return ctypes.util.find_library(name) or f"lib{name}.so"


def _cuda_driver() -> ctypes.CDLL:
    """Load libcuda for the context bookkeeping around driver open."""
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cuCtxGetCurrent.restype = ctypes.c_int
    lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    lib.cuCtxSetCurrent.restype = ctypes.c_int
    return lib


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype on the libopends symbols used here. Idempotent."""
    if getattr(lib.opends_stream_read, "argtypes", None):
        return

    lib.opends_driver_open.argtypes = []
    lib.opends_driver_open.restype = _OpendsError

    lib.opends_driver_close.argtypes = []
    lib.opends_driver_close.restype = _OpendsError

    lib.opends_handle_register.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # opends_handle_t *fh
        ctypes.c_int,  # int fd
    ]
    lib.opends_handle_register.restype = _OpendsError

    lib.opends_handle_deregister.argtypes = [ctypes.c_void_p]  # opends_handle_t fh
    lib.opends_handle_deregister.restype = None  # void

    lib.opends_buf_register.argtypes = [
        ctypes.c_void_p,  # const void *buf_base
        ctypes.c_size_t,  # size_t size
        ctypes.c_int,  # int flags
    ]
    lib.opends_buf_register.restype = _OpendsError

    lib.opends_buf_deregister.argtypes = [ctypes.c_void_p]  # const void *buf_base
    lib.opends_buf_deregister.restype = _OpendsError

    io_args: list[type] = [
        ctypes.c_void_p,  # opends_handle_t fh
        ctypes.c_void_p,  # void *buf_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *buf_offset_p
        ctypes.POINTER(ctypes.c_int64),  # ssize_t *bytes_read_p / bytes_written_p
        ctypes.c_void_p,  # opends_stream_t stream
    ]
    lib.opends_stream_read.argtypes = io_args
    lib.opends_stream_read.restype = _OpendsError
    lib.opends_stream_write.argtypes = io_args
    lib.opends_stream_write.restype = _OpendsError

    lib.opends_stream_register.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    lib.opends_stream_register.restype = _OpendsError

    lib.opends_stream_deregister.argtypes = [ctypes.c_void_p]
    lib.opends_stream_deregister.restype = _OpendsError

    lib.opends_op_status_error.argtypes = [ctypes.c_int]
    lib.opends_op_status_error.restype = ctypes.c_char_p


class Backend(FileGDSBackend):
    """Own the OpenDS driver session and its registration operations."""

    name = "opends"
    _driver = SharedDriver()

    def __init__(self) -> None:
        super().__init__()
        self._init_lock = threading.Lock()
        self._lib_handle: Optional[ctypes.CDLL] = None

    def validate_environment(self) -> None:
        if torch.version.cuda is None:
            raise ValueError("opends requires a CUDA PyTorch build")

    def open_slab(self, location: str, size: int, direct_io: bool) -> "AsyncHandle":
        """Create a slab file; OpenDS registers only ``O_DIRECT`` descriptors."""
        if not direct_io:
            raise ValueError("opends requires --gds-l1-use-direct-io")
        return cast("AsyncHandle", super().open_slab(location, size, direct_io=True))

    def open_handle(self, fd: int, path: str) -> "AsyncHandle":
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return AsyncHandle(self, fd, handle, path)

    def register_handle(self, fd: int) -> int:
        self._ensure_driver_open()
        handle = ctypes.c_void_p()
        self.check_error(
            self.library().opends_handle_register(
                ctypes.byref(handle), ctypes.c_int(fd)
            ),
            "opends_handle_register",
        )
        if handle.value is None:
            raise RuntimeError("opends_handle_register returned a null handle")
        return handle.value

    def deregister_handle(self, handle: int) -> None:
        self.library().opends_handle_deregister(ctypes.c_void_p(handle))

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register a CUDA tensor; aisio exports it as a dma-buf (64 KiB pages)."""
        if not buf.is_cuda:
            raise ValueError("register_buffer: tensor must be on CUDA")
        self._ensure_driver_open()
        nbytes = buf.numel() * buf.element_size()
        self.check_error(
            self.library().opends_buf_register(
                ctypes.c_void_p(buf.data_ptr()),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(0),
            ),
            "opends_buf_register",
        )

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        self.check_error(
            self.library().opends_buf_deregister(ctypes.c_void_p(buf.data_ptr())),
            "opends_buf_deregister",
        )

    def register_stream(self, raw_stream: int) -> None:
        """Use cuFile's FIXED_* flags (0x7) for per-submission IO parameters."""
        self._ensure_driver_open()
        self.check_error(
            self.library().opends_stream_register(
                ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
            ),
            "opends_stream_register",
        )

    def deregister_stream(self, raw_stream: int) -> None:
        self.check_error(
            self.library().opends_stream_deregister(ctypes.c_void_p(raw_stream)),
            "opends_stream_deregister",
        )

    def library(self) -> ctypes.CDLL:
        """Load the OpenDS library once; cached per-DMA lookups skip the lock."""
        if self._lib_handle is not None:
            return self._lib_handle
        with self._init_lock:
            if self._lib_handle is None:
                handle = ctypes.CDLL(_library_path())
                _declare_signatures(handle)
                self._lib_handle = handle
        return self._lib_handle

    def check_error(self, err: "_OpendsError", op: str) -> None:
        """Convert a non-zero ``opends_error_t`` into a Python exception."""
        if err.err != _OPENDS_SUCCESS:
            raise RuntimeError(
                f"{op} failed: opendsError(err={err.err} "
                f"[{self._op_error_string(err.err)}], dev_err={err.dev_err})"
            )

    def _open_driver(self) -> None:
        # The aisio driver brings up its own CUDA context at open and leaves it
        # current on this thread, then destroys it at close. Put the caller's
        # context back so this thread keeps using PyTorch's primary context,
        # from which aisio accepts IO like every other LMCache thread.
        cuda = _cuda_driver()
        context = ctypes.c_void_p()
        cuda.cuCtxGetCurrent(ctypes.byref(context))
        try:
            self.check_error(self.library().opends_driver_open(), "opends_driver_open")
        except Exception:
            # Before cuInit the restore fails too; report the open error.
            cuda.cuCtxSetCurrent(context)
            raise
        rc = cuda.cuCtxSetCurrent(context)
        if rc != 0:
            raise RuntimeError(f"cuCtxSetCurrent failed: CUresult {rc}")

    def _close_driver(self) -> None:
        self.check_error(self.library().opends_driver_close(), "opends_driver_close")

    def _op_error_string(self, err_code: int) -> str:
        """Return the name of an ``opends_op_error_t`` value."""
        raw = self.library().opends_op_status_error(ctypes.c_int(abs(err_code)))
        return raw.decode() if raw is not None else "unknown"


class AsyncHandle(FDGDSHandle):
    """An owning OpenDS slab handle with stream-ordered IO."""

    _backend: Backend

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        self._backend.check_error(
            self._backend.library().opends_stream_read(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "opends_stream_read",
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
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        self._backend.check_error(
            self._backend.library().opends_stream_write(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.result),
                ctypes.c_void_p(raw_stream),
            ),
            "opends_stream_write",
        )
        return sub
