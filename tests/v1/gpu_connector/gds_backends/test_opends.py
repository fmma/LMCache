# SPDX-License-Identifier: Apache-2.0
"""OpenDS tests with a fake native library."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
import ctypes
import ctypes.util
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends import opends as oa


def _ok() -> oa._OpendsError:
    return oa._OpendsError(err=oa._OPENDS_SUCCESS, dev_err=0)


def _err(code: int = 5001, dev_err: int = 0) -> oa._OpendsError:
    return oa._OpendsError(err=code, dev_err=dev_err)


class _FakeLib:
    """Stand-in for ``libopends_*.so`` that records all symbol calls."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {}

    def __getattr__(self, name: str) -> Any:
        def _record(*args: Any) -> Any:
            self.calls.setdefault(name, []).append(args)
            if name == "opends_op_status_error":
                return b"opends fake error"
            if name == "opends_handle_deregister":
                return None
            if name == "opends_handle_register":
                args[0]._obj.value = 0xFEED
            if name in ("opends_stream_read", "opends_stream_write"):
                args[5]._obj.value = args[2]._obj.value
            return _ok()

        return _record


@pytest.fixture
def backend() -> oa.Backend:
    return oa.Backend()


class _FakeCuda:
    """Stand-in for libcuda recording the context bound to the opening thread."""

    def __init__(self, current: int = 0xC0FFEE) -> None:
        self.current = current
        self.calls: list[tuple[str, int]] = []

    def cuCtxGetCurrent(self, ref: Any) -> int:
        ref._obj.value = self.current
        self.calls.append(("get", self.current))
        return 0

    def cuCtxSetCurrent(self, context: Any) -> int:
        self.calls.append(("set", context.value or 0))
        self.current = context.value or 0
        return 0


@pytest.fixture
def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> _FakeCuda:
    cuda = _FakeCuda()
    monkeypatch.setattr(oa, "_cuda_driver", lambda: cuda)
    return cuda


@pytest.fixture(autouse=True)
def _fake_lib(
    backend: oa.Backend, _fake_cuda: _FakeCuda, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_FakeLib]:
    """Replace the native library and libcuda with fakes."""
    lib = _FakeLib()
    monkeypatch.setattr(backend, "library", lambda: lib)
    yield lib
    backend.close_driver()


def _fake_gpu_tensor(ptr: int = 0x1000, nbytes: int = 4096) -> SimpleNamespace:
    """Return a GPU-tensor stand-in accepted by the wrapper."""
    return SimpleNamespace(
        is_cuda=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
    )


class TestEnvironment:
    def test_requires_cuda_build(
        self, backend: oa.Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.version, "cuda", None)
        with pytest.raises(ValueError, match="CUDA"):
            backend.validate_environment()

    def test_is_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.version, "cuda", "12.9")
        assert not oa.Backend.is_default()


class TestLibraryLoading:
    @pytest.mark.parametrize(
        ("environ", "found", "expected", "searched"),
        [
            ({}, None, "libopends_aisio.so", ["opends_aisio"]),
            (
                {},
                "/opt/lib/libopends_aisio.so.0",
                "/opt/lib/libopends_aisio.so.0",
                ["opends_aisio"],
            ),
            (
                {"OPENDS_BACKEND": "cufile"},
                None,
                "libopends_cufile.so",
                ["opends_cufile"],
            ),
            (
                {"OPENDS_LIBRARY": "/build/libopends_ref.so"},
                None,
                "/build/libopends_ref.so",
                [],
            ),
        ],
    )
    def test_library_path_follows_opends_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        environ: dict[str, str],
        found: str | None,
        expected: str,
        searched: list[str],
    ) -> None:
        monkeypatch.delenv("OPENDS_BACKEND", raising=False)
        monkeypatch.delenv("OPENDS_LIBRARY", raising=False)
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
        names: list[str] = []

        def find_library(name: str) -> str | None:
            names.append(name)
            return found

        monkeypatch.setattr(ctypes.util, "find_library", find_library)
        assert oa._library_path() == expected
        assert names == searched

    def test_library_loads_once_and_declares_signatures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = oa.Backend()
        loaded: list[str] = []

        class _Symbol:
            argtypes: object = None
            restype: object = None

        class _Lib:
            def __init__(self, path: str) -> None:
                loaded.append(path)
                self.symbols: dict[str, _Symbol] = {}

            def __getattr__(self, name: str) -> _Symbol:
                return self.symbols.setdefault(name, _Symbol())

        monkeypatch.setattr(oa.ctypes, "CDLL", _Lib)
        monkeypatch.setattr(oa, "_library_path", lambda: "libopends_fake.so")
        lib = backend.library()
        assert backend.library() is lib
        assert loaded == ["libopends_fake.so"]
        assert lib.opends_stream_read.restype is oa._OpendsError
        assert lib.opends_stream_write.argtypes == lib.opends_stream_read.argtypes
        assert lib.opends_handle_register.argtypes == [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        assert lib.opends_op_status_error.restype is ctypes.c_char_p


class TestDriverLifecycle:
    def test_open_restores_the_callers_cuda_context(
        self, backend: oa.Backend, _fake_lib: _FakeLib, _fake_cuda: _FakeCuda
    ) -> None:
        def _open() -> oa._OpendsError:
            # The native driver leaves its own context current on this thread.
            _fake_cuda.current = 0xDEAD
            _fake_cuda.calls.append(("open", 0xDEAD))
            return _ok()

        _fake_lib.opends_driver_open = _open
        backend.register_stream(7)
        backend.register_stream(9)
        backend.register_buffer(_fake_gpu_tensor())
        assert _fake_cuda.calls == [
            ("get", 0xC0FFEE),
            ("open", 0xDEAD),
            ("set", 0xC0FFEE),
        ]
        assert _fake_cuda.current == 0xC0FFEE

    def test_failed_open_still_restores_the_context(
        self, backend: oa.Backend, _fake_lib: _FakeLib, _fake_cuda: _FakeCuda
    ) -> None:
        _fake_cuda.current = 0
        _fake_lib.opends_driver_open = lambda: _err(5033)
        with pytest.raises(RuntimeError, match="opends_driver_open"):
            backend.register_stream(7)
        assert _fake_cuda.calls == [("get", 0), ("set", 0)]

    def test_failed_open_wins_over_a_failed_restore(
        self,
        backend: oa.Backend,
        _fake_lib: _FakeLib,
        _fake_cuda: _FakeCuda,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_cuda.current = 0
        _fake_lib.opends_driver_open = lambda: _err(5033)
        restore = _fake_cuda.cuCtxSetCurrent

        def restore_before_cuinit(context: Any) -> int:
            restore(context)
            return 3  # CUDA_ERROR_NOT_INITIALIZED

        monkeypatch.setattr(_fake_cuda, "cuCtxSetCurrent", restore_before_cuinit)
        with pytest.raises(RuntimeError, match="opends_driver_open"):
            backend.register_stream(7)
        assert _fake_cuda.calls == [("get", 0), ("set", 0)]

    def test_context_restore_failure_raises(
        self,
        backend: oa.Backend,
        _fake_cuda: _FakeCuda,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_fake_cuda, "cuCtxSetCurrent", lambda context: 201)
        with pytest.raises(RuntimeError, match="cuCtxSetCurrent.*201"):
            backend.register_stream(7)

    def test_overlapping_backends_share_driver(
        self, backend: oa.Backend, _fake_lib: _FakeLib, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = oa.Backend()
        monkeypatch.setattr(other, "library", lambda: _fake_lib)
        try:
            backend.register_stream(7)
            other.register_stream(9)
            handle = other.open_handle(-1, "/slab")
            assert _fake_lib.calls["opends_driver_open"] == [()]
            backend.deregister_stream(7)
            backend.close_driver()
            assert "opends_driver_close" not in _fake_lib.calls
            other.register_buffer(_fake_gpu_tensor())
            handle.read_async(0x1000, 4096, 0, 0, 9)
            assert len(_fake_lib.calls["opends_stream_read"]) == 1
            other.deregister_buffer(_fake_gpu_tensor())
            other.deregister_handle(0xFEED)
            other.deregister_stream(9)
        finally:
            backend.close_driver()
            other.close_driver()
        assert _fake_lib.calls["opends_driver_close"] == [()]

    def test_native_open_error_propagates(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.opends_driver_open = lambda: _err(5033)
        with pytest.raises(RuntimeError, match="opends_driver_open.*5033"):
            backend.register_stream(0)
        backend.close_driver()
        assert "opends_driver_close" not in _fake_lib.calls

    def test_native_close_error_propagates(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_stream(0)
        _fake_lib.opends_driver_close = lambda: _err(5006)
        with pytest.raises(RuntimeError, match="opends_driver_close"):
            backend.close_driver()


class TestHandleRegistration:
    def test_returns_native_handle_for_fd(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        captured: dict[str, int] = {}

        def _register(handle_ref: Any, fd: Any) -> oa._OpendsError:
            captured["fd"] = fd.value
            handle_ref._obj.value = 0xDEADBEEF
            return _ok()

        _fake_lib.opends_handle_register = _register
        handle = backend.register_handle(42)
        assert handle == 0xDEADBEEF
        assert captured == {"fd": 42}
        backend.deregister_handle(handle)
        (native_handle,) = _fake_lib.calls["opends_handle_deregister"][0]
        assert native_handle.value == handle

    def test_rejects_null_handle(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.opends_handle_register = lambda *args: _ok()
        with pytest.raises(RuntimeError, match="null handle"):
            backend.register_handle(7)

    def test_error_names_status(self, backend: oa.Backend, _fake_lib: _FakeLib) -> None:
        _fake_lib.opends_handle_register = lambda *args: _err(5020)
        with pytest.raises(RuntimeError) as exc:
            backend.register_handle(7)
        assert "opends_handle_register" in str(exc.value)
        assert "5020" in str(exc.value)
        assert "opends fake error" in str(exc.value)
        (status,) = _fake_lib.calls["opends_op_status_error"][0]
        assert status.value == 5020


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self, backend: oa.Backend) -> None:
        with pytest.raises(ValueError, match="CUDA"):
            backend.register_buffer(SimpleNamespace(is_cuda=False))  # type: ignore[arg-type]

    def test_passes_pointer_size_and_flags(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_buffer(_fake_gpu_tensor(ptr=0x2000, nbytes=8192))
        pointer, length, flags = _fake_lib.calls["opends_buf_register"][0]
        assert pointer.value == 0x2000
        assert length.value == 8192
        assert flags.value == 0
        backend.deregister_buffer(_fake_gpu_tensor(ptr=0x2000))
        (pointer,) = _fake_lib.calls["opends_buf_deregister"][0]
        assert pointer.value == 0x2000

    def test_register_error_propagates(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.opends_buf_register = lambda *args: _err(5011, dev_err=-1)
        with pytest.raises(RuntimeError, match="opends_buf_register.*5011.*-1"):
            backend.register_buffer(_fake_gpu_tensor())

    def test_deregister_error_propagates(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.opends_buf_deregister = lambda *args: _err(5024)
        with pytest.raises(RuntimeError, match="opends_buf_deregister"):
            backend.deregister_buffer(_fake_gpu_tensor())


class TestStreamRegistration:
    def test_register_stream_uses_fixed_flags(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        backend.register_stream(0xABC)
        stream, flags = _fake_lib.calls["opends_stream_register"][0]
        assert stream.value == 0xABC
        # FIXED_BUF_OFFSET | FIXED_FILE_OFFSET | FIXED_FILE_SIZE.
        assert flags == 0x7
        backend.deregister_stream(0xABC)
        (stream,) = _fake_lib.calls["opends_stream_deregister"][0]
        assert stream.value == 0xABC

    def test_register_stream_propagates_error(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.opends_stream_register = lambda *args: _err(5030)
        with pytest.raises(RuntimeError, match="opends_stream_register"):
            backend.register_stream(0xABC)


class TestAsyncHandleIO:
    def _handle(self, backend: oa.Backend) -> oa.AsyncHandle:
        return oa.AsyncHandle(backend=backend, fd=5, handle=0xFEED, path="/slab")

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_retains_native_argument_storage(
        self, backend: oa.Backend, _fake_lib: _FakeLib, operation: str
    ) -> None:
        handle = self._handle(backend)
        submit = handle.read_async if operation == "read" else handle.write_async
        sub = submit(
            buf_base=0x3000, size=4096, file_offset=8192, buf_offset=512, raw_stream=0x9
        )
        calls = _fake_lib.calls[f"opends_stream_{operation}"]
        assert len(calls) == 1
        fh, buf, size, offset, buf_offset, result, stream = calls[0]
        assert fh.value == 0xFEED
        assert buf.value == 0x3000
        assert stream.value == 0x9
        assert size._obj is sub.size and size._obj.value == 4096
        assert offset._obj is sub.file_offset and offset._obj.value == 8192
        assert buf_offset._obj is sub.buf_offset and buf_offset._obj.value == 512
        assert result._obj is sub.result
        assert sub.bytes_done == 4096

    def test_submission_error_raises(
        self, backend: oa.Backend, _fake_lib: _FakeLib
    ) -> None:
        _fake_lib.opends_stream_write = lambda *args: _err(5015, dev_err=201)
        with pytest.raises(RuntimeError, match="opends_stream_write.*5015.*201"):
            self._handle(backend).write_async(
                buf_base=0x3000, size=2048, file_offset=0, buf_offset=0, raw_stream=0x9
            )

    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_deferred_error_is_reported_in_result(
        self, backend: oa.Backend, _fake_lib: _FakeLib, operation: str
    ) -> None:
        def _submit(*args: Any) -> oa._OpendsError:
            args[5]._obj.value = -5022
            return _ok()

        setattr(_fake_lib, f"opends_stream_{operation}", _submit)
        handle = self._handle(backend)
        submit = handle.read_async if operation == "read" else handle.write_async
        sub = submit(
            buf_base=0x3000, size=4096, file_offset=0, buf_offset=0, raw_stream=9
        )
        assert sub.bytes_done == -5022


class TestSlab:
    def test_rejects_buffered_io(self, backend: oa.Backend) -> None:
        with pytest.raises(ValueError, match="direct-io"):
            backend.open_slab("/slabs", 4096, direct_io=False)

    def test_opens_slab_file_with_direct_io(
        self, backend: oa.Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[object] = []

        def open_file(path: str, flags: int, *args: object) -> int:
            calls.append(("open", path, flags))
            return 7

        monkeypatch.setattr(os, "makedirs", lambda *a, **kw: calls.append("mkdir"))
        monkeypatch.setattr(os, "open", open_file)
        monkeypatch.setattr(os, "close", lambda fd: calls.append(("close", fd)))
        monkeypatch.setattr(
            os,
            "posix_fallocate",
            lambda *a: calls.append(("allocate", a)),
            raising=False,
        )
        monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
        handle = backend.open_slab("/slabs", 4096, direct_io=True)
        assert handle.path == "/slabs/lmcache_gds_slab.bin"
        assert calls == [
            "mkdir",
            ("open", handle.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC),
            ("allocate", (7, 0, 4096)),
            ("close", 7),
            ("open", handle.path, os.O_RDWR | os.O_DIRECT),
        ]
        handle.close()
        assert calls[-1] == ("close", 7)


class TestStructLayout:
    """Guard the ctypes struct against ``opends.h`` on LP64 platforms."""

    def test_error_struct_layout(self) -> None:
        assert ctypes.sizeof(oa._OpendsError) == 8
        assert oa._OpendsError.err.offset == 0
        assert oa._OpendsError.dev_err.offset == 4
