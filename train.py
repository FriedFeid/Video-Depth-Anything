import torch
from time import sleep
import random
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--vram", "-v", type=float, default=38.0)
    parser.add_argument("--compute", "-c", action="store_true")
    parser.add_argument("--gpus", "-g", type=int, nargs="+", default=[0])

    args = parser.parse_args()
    
    return args



def allocate_vram(size_in_gb, devices):
    """
    Allocate a specific amount of VRAM on the GPU.
    
    Args:
        size_in_gb (float): The amount of VRAM to allocate in gigabytes.
    """
    # Convert the size from GB to bytes
    size_in_bytes = size_in_gb * 1024**3

    # Calculate the number of elements needed for the tensor
    dtype = torch.float32  # You can choose other dtypes if needed
    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(size_in_bytes // element_size)

    # Allocate the tensor on the GPU
    for d in devices:
        device = torch.device(f"cuda:{d}")
        tensor = torch.zeros(num_elements, dtype=dtype, device=device)

    print(f"Allocated {tensor.nelement() * element_size / 1024**3:.2f} GB of VRAM on {device}")

def stress_gpu(devices):
    """ Perform useless calculations to stress the GPU. """

    device = torch.device(f'cuda:{devices[0]}')
    size = 1024  # Size of the matrix for multiplication
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    print("Starting GPU stress test...")
    i = 0
    while True:
        _ = torch.mm(A, B)  # Matrix multiplication
        if i % 5000 == 0:
            time = random.uniform(0.1, 0.4)
            sleep(time)
        i += 1
         
    print("Completed GPU stress test.")

if __name__ == '__main__':
    args = parse_args()
    allocate_vram(args.vram, args.gpus)
    while True:
        if args.compute:
            stress_gpu(args.gpus)
        #sleep(1)

