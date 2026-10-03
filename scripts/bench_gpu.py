"""Small CUDA benchmark for the local ControlNet runtime."""

import time

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    props = torch.cuda.get_device_properties(0)
    print("=== GPU Info ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"FP32 Matmul Allow TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"Cudnn Allow TF32: {torch.backends.cudnn.allow_tf32}")
    print()
    print("=== FP16 Matmul Benchmark (10240x10240) ===")

    a = torch.randn(10240, 10240, device="cuda", dtype=torch.float16)
    b = torch.randn(10240, 10240, device="cuda", dtype=torch.float16)
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()
    avg_ms = (time.time() - start) / 10 * 1000
    tflops = 10240**3 * 2 / (avg_ms / 1000) / 1e12
    print(f"Avg time: {avg_ms:.2f} ms, throughput: {tflops:.1f} TFLOPS")
    print(f"Driver reports multi_gpu_board: {getattr(props, 'multi_gpu_board', 'unknown')}")


if __name__ == "__main__":
    main()
