# Test made with AI to speed up validation, same style as test_agent.py /
# test_benchmark.py: plain asserts + prints, no pytest, fully offline against
# a fake filesystem tree (no real Ollama/LM Studio/HF install required).
import os
import shutil
import tempfile
import time

from llmadapt.benchmark import BenchmarkResult
from llmadapt.hardware import ResourceQuota
from llmadapt.router import ModelRouter
from llmadapt.selector import ModelCatalog, LocalModelSelector


def make_benchmark(gpu_vram_gb, system_ram_gb, cpu_gflops=100.0, pcie_bandwidth_gbps=8.0, gpu_tflops=20.0):
    gpu_name = "none" if gpu_vram_gb <= 0 else "Fake GPU"
    return BenchmarkResult(
        os="Linux", cpu_name="Fake CPU", cpu_cores=8,
        cpu_gflops=cpu_gflops, cpu_gflops_measured_with="numpy",
        system_ram_gb=system_ram_gb,
        gpu_name=gpu_name, gpu_vram_gb=gpu_vram_gb,
        gpu_pcie_gen=4, gpu_pcie_link_width=16,
        pcie_bandwidth_gbps=pcie_bandwidth_gbps, pcie_bandwidth_measured=False,
        gpu_tflops=gpu_tflops, gpu_tflops_measured=False,
        fingerprint="test", timestamp=time.time(),
    )


# The selector only ever learns a model's size through os.path.getsize. This
# test used to create real multi-GB files with f.truncate() - free on Linux
# (sparse files) but on Windows/NTFS it allocated ~64GB across the fake models
# and hung, which is why run_all.py had to skip the whole file. Now every
# "weights" file is a few bytes on disk and its pretend size is served from
# FAKE_SIZES by a patched getsize, so the test is instant on every OS.
FAKE_SIZES = {}
_real_getsize = os.path.getsize


def _fake_getsize(path):
    key = os.path.normcase(os.path.abspath(path))
    if key in FAKE_SIZES:
        return FAKE_SIZES[key]
    return _real_getsize(path)


os.path.getsize = _fake_getsize


def touch_sized_file(path, size_bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")
    FAKE_SIZES[os.path.normcase(os.path.abspath(path))] = size_bytes


root = tempfile.mkdtemp(prefix="llmadapt_selector_test_")
GB = 1024 ** 3

try:
    # ---- build a fake Ollama install: llama3.1:8b, ~4GB blob ----
    ollama_base = os.path.join(root, "ollama")
    manifest_dir = os.path.join(ollama_base, "manifests", "registry.ollama.ai", "library", "llama3.1")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "8b"), "w") as f:
        f.write('{"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc123"}]}')
    touch_sized_file(os.path.join(ollama_base, "blobs", "sha256-abc123"), 4 * GB)

    # ---- a second, bigger Ollama model: llama3.1:70b, ~40GB blob ----
    manifest_dir_70b = manifest_dir  # same model family, different tag
    with open(os.path.join(manifest_dir_70b, "70b"), "w") as f:
        f.write('{"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:def456"}]}')
    touch_sized_file(os.path.join(ollama_base, "blobs", "sha256-def456"), 40 * GB)

    # ---- fake LM Studio install: one gguf, ~6GB ----
    lmstudio_base = os.path.join(root, "lmstudio")
    gguf_path = os.path.join(lmstudio_base, "bartowski", "Meta-Llama-3-8B-Instruct-GGUF", "model.Q4_K_M.gguf")
    touch_sized_file(gguf_path, 6 * GB)

    # ---- fake HF cache: one repo, two safetensors shards summing to ~14GB ----
    hf_base = os.path.join(root, "hf")
    snap_dir = os.path.join(hf_base, "models--mistralai--Mistral-7B-v0.1", "snapshots", "abcdef")
    touch_sized_file(os.path.join(snap_dir, "model-00001-of-00002.safetensors"), 7 * GB)
    touch_sized_file(os.path.join(snap_dir, "model-00002-of-00002.safetensors"), 7 * GB)

    base_paths = {"ollama": ollama_base, "lm-studio": lmstudio_base, "hf": hf_base}
    catalog = ModelCatalog(base_paths=base_paths)

    # Test 1: discovery finds all three providers with roughly correct sizes
    found = catalog.list_candidates(include_uninstalled=False)
    by_name = {c.name: c for c in found}
    assert "llama3.1:8b" in by_name and by_name["llama3.1:8b"].provider == "ollama"
    assert abs(by_name["llama3.1:8b"].size_gb - 4.0) < 0.01
    assert "llama3.1:70b" in by_name
    assert abs(by_name["llama3.1:70b"].size_gb - 40.0) < 0.01
    assert "bartowski/Meta-Llama-3-8B-Instruct-GGUF/model.Q4_K_M.gguf" in by_name
    assert "mistralai/Mistral-7B-v0.1" in by_name
    assert abs(by_name["mistralai/Mistral-7B-v0.1"].size_gb - 14.0) < 0.01
    print("PASS: discovery finds Ollama, LM Studio, and HF candidates with correct sizes")

    # Test 2: register() with no installed/discovered match -> not installed, has an install hint
    ghost = catalog.register("qwen3-moe:235b", provider="ollama")
    assert ghost.installed is False
    assert "ollama pull qwen3-moe:235b" in ghost.install_hint
    print("PASS: registering an uninstalled model returns installed=False with an install hint")

    # Test 3: register() with a name that IS discovered adopts the real entry, not a stale guess
    adopted = catalog.register("llama3.1:8b", provider="ollama", size_gb=999.0)
    assert adopted.installed is True
    assert abs(adopted.size_gb - 4.0) < 0.01, "registration should adopt the real discovered size, not the hand-typed one"
    print("PASS: registering a name that's actually installed adopts its real discovered size")

    # Test 4: find() locates by name (+ optional provider)
    assert catalog.find("llama3.1:8b") is not None
    assert catalog.find("llama3.1:8b", provider="lm-studio") is None
    print("PASS: ModelCatalog.find() respects the provider filter")

    # ---- selection scoring ----

    # Test 5: plenty of VRAM -> the bigger model wins, tier gpu_resident
    big_vram_bench = make_benchmark(gpu_vram_gb=48.0, system_ram_gb=64.0)
    result = LocalModelSelector.select_best(big_vram_bench, catalog)
    assert result.chosen is not None
    assert result.chosen.name == "llama3.1:70b", f"expected the biggest fully-resident model, got {result.chosen.name}"
    assert result.ranked[0].tier == "gpu_resident"
    print("PASS: with plenty of VRAM, auto mode picks the biggest model that's still fully GPU-resident")

    # Test 6: small VRAM but big RAM -> falls back to a smaller/offloaded pick, not infeasible
    small_vram_bench = make_benchmark(gpu_vram_gb=6.0, system_ram_gb=64.0, cpu_gflops=250.0, pcie_bandwidth_gbps=20.0)
    result2 = LocalModelSelector.select_best(small_vram_bench, catalog)
    assert result2.chosen is not None
    assert result2.chosen.name != "llama3.1:70b" or result2.ranked[0].tier == "offloaded"
    print(f"PASS: with limited VRAM, auto mode still finds a workable pick ({result2.chosen.name}, tier={result2.ranked[0].tier})")

    # Test 7: tiny VRAM and RAM -> nothing fits, chosen is None with a clear reason
    tiny_bench = make_benchmark(gpu_vram_gb=2.0, system_ram_gb=4.0)
    result3 = LocalModelSelector.select_best(tiny_bench, catalog)
    assert result3.chosen is None
    assert "fits" in result3.reason or "install" in result3.reason.lower()
    print("PASS: when nothing fits this machine, chosen is None with an explanatory reason")

    # Test 8: requested_name for an installed model returns it directly
    result4 = LocalModelSelector.select_best(big_vram_bench, catalog, requested_name="llama3.1:8b")
    assert result4.chosen is not None and result4.chosen.name == "llama3.1:8b"
    print("PASS: requested_name pins selection to that specific installed model")

    # Test 9: requested_name for a registered-but-uninstalled model surfaces install guidance
    result5 = LocalModelSelector.select_best(big_vram_bench, catalog, requested_name="qwen3-moe:235b", requested_provider="ollama")
    assert result5.chosen is None
    assert result5.needs_install is not None and result5.needs_install.installed is False
    assert "ollama pull qwen3-moe:235b" in result5.install_hint
    print("PASS: requesting a registered-but-uninstalled model returns install guidance, not a silent fallback")

    # Test 10: requested_name for a name never seen anywhere also surfaces install guidance
    result6 = LocalModelSelector.select_best(big_vram_bench, catalog, requested_name="totally-unknown-model:1b")
    assert result6.chosen is None
    assert result6.needs_install is not None
    assert result6.needs_install.name == "totally-unknown-model:1b"
    print("PASS: requesting a completely unknown model name still returns actionable install guidance")

    # Test 11: nothing installed at all, but something registered -> points at it instead of failing flat
    empty_catalog = ModelCatalog(base_paths={"ollama": os.path.join(root, "empty"), "lm-studio": os.path.join(root, "empty"), "hf": os.path.join(root, "empty")})
    empty_catalog.register("llama3.1:8b", provider="ollama")
    result7 = LocalModelSelector.select_best(big_vram_bench, empty_catalog)
    assert result7.chosen is None
    assert result7.needs_install is not None and result7.needs_install.name == "llama3.1:8b"
    print("PASS: with nothing installed, auto mode points at a registered-but-missing model instead of just failing")

    # Test 12: router integration exposes the same behavior
    router_result = ModelRouter.allocate_local_auto(big_vram_bench, catalog=catalog)
    assert router_result.chosen is not None and router_result.chosen.name == "llama3.1:70b"
    print("PASS: ModelRouter.allocate_local_auto() reaches the same selection through the router")

    # Test 13: manual ResourceQuota overrides the benchmark's own capacity numbers
    quota = ResourceQuota(mode="manual", max_vram_gb=6.0, max_ram_gb=64.0)
    result8 = LocalModelSelector.select_best(big_vram_bench, catalog, quota=quota)
    assert result8.chosen is not None and result8.chosen.name != "llama3.1:70b", (
        "a manual 6GB VRAM quota should override the 48GB benchmark reading"
    )
    print("PASS: a manual ResourceQuota overrides the benchmark's measured VRAM/RAM")

    print("\nAll checks passed.")
finally:
    shutil.rmtree(root, ignore_errors=True)
