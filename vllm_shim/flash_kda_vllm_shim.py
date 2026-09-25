"""Routes GLM-5.3-Flash's chunked KDA prefill in vLLM through this repo's
flash_kda wheel. vLLM otherwise calls its own compiled copy, vllm._flashkda_C.
flash_kda_vllm_shim.pth loads this from site-packages when FLASH_KDA_SHIM=1.
The call keeps vLLM's arguments and workspace buffers, so only the kernel
changes. It needs a vLLM nightly from 2026-09-14 or later, where
Glm5NextLinearAttention._flashkda_prefill exists with this signature."""
import os

if os.environ.get("FLASH_KDA_SHIM", "0") == "1":
    import importlib.abc
    import importlib.util
    import sys

    TARGET = "vllm.models.glm5next.common.kda"

    def _patch(mod):
        import torch
        import flash_kda  # noqa: F401  registers torch.ops.flash_kda

        cls = getattr(mod, "Glm5NextLinearAttention", None)
        if cls is None or getattr(cls, "_kfix_patched", False):
            return
        wm = mod.current_workspace_manager

        called = {"n": 0}

        def _flashkda_prefill(self, q, k, v, g, beta, initial_state, cu_seqlens, out):
            if called["n"] == 0:
                print("[flash_kda shim] first prefill call: the wheel's kernel is running", file=sys.stderr)
            called["n"] += 1
            assert self._flashkda_buffer_specs is not None
            final_state, workspace, workspace_out = wm().get_simultaneous(*self._flashkda_buffer_specs)
            final_state = final_state[: initial_state.shape[0]]
            if out is None:
                out = workspace_out[:, : q.shape[1]]
            torch.ops.flash_kda.fwd(
                q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta,
                self.head_dim ** -0.5, out, workspace, self.A_log.view(-1),
                self.dt_bias.view(-1, self.head_dim), self.kda_lower_bound,
                initial_state.contiguous(), final_state, cu_seqlens.contiguous(), None, None)
            return out, final_state

        cls._flashkda_prefill = _flashkda_prefill
        cls._kfix_patched = True
        print("[flash_kda shim] installed; a second line follows on the first prefill call", file=sys.stderr)

    class _Finder(importlib.abc.MetaPathFinder):
        _busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET or _Finder._busy:
                return None
            _Finder._busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                _Finder._busy = False
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec_):
                    return loader.create_module(spec_) if hasattr(loader, "create_module") else None

                def exec_module(self, module):
                    loader.exec_module(module)
                    _patch(module)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Finder())
