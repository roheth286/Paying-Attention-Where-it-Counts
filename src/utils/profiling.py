import time
import torch
from thop import profile

def profile_model(model, input_size=(1, 3, 256, 256)):
    """
    Counts parameters and MACs (FLOPs) using the thop package.
    """
    x = torch.randn(*input_size)
    macs, params = profile(model, inputs=(x,))
    print(f"MACs (Operations): {macs}")
    print(f"Parameters count: {params}")
    return macs, params


def print_layer_parameters(model):
    """
    Prints parameter count for each top-level child layer of the model.
    """
    print("\nLayer-wise Parameter Breakdown:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {param_count}")


def benchmark_latency(model, device, input_size=(1, 3, 256, 256), runs=50, warmup=10):
    """
    Benchmarks the model's forward pass latency on the specified device.
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(runs):
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        end_time = time.time()

    avg_inference_time = (end_time - start_time) / runs
    print(f"Avg inference time: {avg_inference_time * 1000:.2f} ms")
    return avg_inference_time
