#!/usr/bin/env python3
import concurrent.futures
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_EXEC = ["uv", "run", "ipython"]
NB_NAME = "solutions/05_Benchmarking.ipynb"
NUM_PROCESSES = 10
TIMEOUT = 80 * 30


def run_test_process(test_id, notebook_name):
    env = os.environ.copy()
    env["TEST_ID"] = str(test_id) + str(time.time_ns())

    start_time = time.time()

    try:
        result = subprocess.run(
            SCRIPT_EXEC + [notebook_name],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"Process {test_id} completed successfully in {duration:.2f}s")
            return {
                "test_id": test_id,
                "success": True,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        else:
            print(f"Process {test_id} failed with exit code {result.returncode}")
            return {
                "test_id": test_id,
                "success": False,
                "duration": duration,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
    except subprocess.TimeoutExpired as e:
        print(f"Process {test_id} timed out after 5 minutes")
        return {
            "test_id": test_id,
            "success": False,
            "duration": TIMEOUT,
            "error": "timeout",
            "stdout": e.stdout.decode("utf-8") if e.stdout is not None else "",
            "stderr": e.stderr.decode("utf-8") if e.stderr is not None else "",
    }
    except Exception as e:
        print(f"Process {test_id} failed with error: {e}")
        return {"test_id": test_id, "success": False, "duration": 0, "error": str(e)}


def main():
    print(f"Starting load test with {NUM_PROCESSES} processes...")
    print(f"Script: {NB_NAME}")
    print(f"Started at: {datetime.now()}")

    start_time = time.time()

    # Run all processes concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # Submit all tasks
        futures = [
            executor.submit(run_test_process, i + 1, NB_NAME)
            for i in range(NUM_PROCESSES)
        ]

        # Collect results as they complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    # Calculate summary statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    failed = NUM_PROCESSES - successful

    if successful > 0:
        avg_duration = sum(r["duration"] for r in results if r["success"]) / successful
    else:
        avg_duration = 0

    # Print summary
    print("\n" + "=" * 50)
    print("LOAD TEST SUMMARY")
    print("=" * 50)
    print(f"Total processes: {NUM_PROCESSES}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful / NUM_PROCESSES) * 100:.1f}%")
    print(f"Average process duration: {avg_duration:.2f}s")
    print(f"Total test duration: {total_time:.2f}s")
    print(f"Completed at: {datetime.now()}")

    # Print failed processes details
    if failed > 0:
        print("\nFAILED PROCESSES:")
        for result in results:
            if not result["success"]:
                print(
                    f"  Process {result['test_id']}: {result.get('error', 'exit code ' + str(result.get('exit_code', 'unknown')))}"
                )
    for result in results:
        with open(f"logs/{result['test_id']}", "w") as f:
            f.write(result.get("stdout", ""))
            f.write("="*50 + "\n")
            f.write(result.get("stderr", ""))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
