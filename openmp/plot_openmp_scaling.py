import matplotlib.pyplot as plt

threads = [1, 2, 4, 8, 16]

time_no_tasks = [
    119.198,  # 1 thread: 119198.446 ms
    72.144,   # 2 threads: 72144.201 ms
    52.936,   # 4 threads: 52935.530 ms
    43.090,   # 8 threads: 43090.406 ms
    72.416,   # 16 threads: 72416.303 ms
]

time_tasks = [
    127.209,  # 1 thread: 127209.313 ms
    63.876,   # 2 threads: 63876.120 ms
    32.837,   # 4 threads: 32837.497 ms
    17.937,   # 8 threads: 17937.396 ms
    11.291,   # 16 threads: 11290.866 ms
]

def main():
    plt.figure()
    plt.plot(threads, time_no_tasks, "o-", label="OpenMP (no tasks)")
    plt.plot(threads, time_tasks, "s-", label="OpenMP tasks (batch-parallel)")

    plt.xlabel("Number of OpenMP threads")
    plt.ylabel("Training time (s)")
    plt.title("OpenMP Scaling of MLP Training (10k samples, 2000 passes)")
    plt.xticks(threads)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # Save both vector and bitmap versions
    plt.savefig("openmp_scaling.pdf")
    plt.savefig("openmp_scaling.png", dpi=300)

if __name__ == "__main__":
    main()
