import os

import ray


@ray.remote
def hello_worker() -> str:
    return f"Hola desde worker PID={os.getpid()}"


def test_ray_workers() -> None:
    ray.init(ignore_reinit_error=True)
    futures = [hello_worker.remote() for _ in range(4)]
    results = ray.get(futures)
    assert len(results) == 4
    ray.shutdown()


if __name__ == "__main__":
    test_ray_workers()
