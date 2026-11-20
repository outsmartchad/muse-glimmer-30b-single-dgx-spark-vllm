FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

ENV VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=75903e506743180c391bc01ad9bf06a1bab68048 \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    MAX_JOBS=4

RUN python3 -m venv /opt/muse-venv \
    && git init /opt/muse-vllm \
    && git -C /opt/muse-vllm fetch --depth=1 https://github.com/xianbaoqian/vllm.git 99a10304dce8945119bd0b1a072297803c52a749 \
    && git -C /opt/muse-vllm checkout --detach FETCH_HEAD \
    && /opt/muse-venv/bin/python -m pip install ninja -e /opt/muse-vllm

ENV PATH=/opt/muse-venv/bin:$PATH
