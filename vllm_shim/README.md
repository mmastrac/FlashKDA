# Trying this branch inside a vLLM container

vLLM calls the copy of FlashKDA compiled into it, `vllm._flashkda_C`. The shim
here points GLM-5.3-Flash's chunked prefill at this wheel's op. Decode keeps
its own recurrent kernel in both cases.

Inside the container (it needs `nvcc` and the container's own torch):

    git clone -b wip-fp32-state --recurse-submodules https://github.com/mmastrac/FlashKDA.git
    cd FlashKDA
    FLASH_KDA_CUDA_ARCHS=120f pip wheel . -w dist --no-build-isolation --no-deps   # 90a, 100a, 103a or 120a for other GPUs
    pip install --no-deps dist/flash_kda-*.whl
    SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
    cp vllm_shim/flash_kda_vllm_shim.py vllm_shim/flash_kda_vllm_shim.pth "$SITE"/

Then start vLLM with `FLASH_KDA_SHIM=1` in the environment on every rank. The
engine log prints `[flash_kda shim] installed` once per process at import, and
`[flash_kda shim] first prefill call: the wheel's kernel is running` the first
time a prefill goes through it. Only the second line proves the wheel ran: if
`kda_prefill_backend` resolved to `triton`, the first line still appears and
the second never does.

To check the kernel alone before serving:

    python -m pytest -q tests/test_fwd.py -k "test_fwd and not fla"

The build takes about five minutes on a GB10 and the two tests about five more.
`tests/test_fwd.py -k fla` also needs the `flash-linear-attention` package.
