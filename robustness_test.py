import subprocess
import time

def run_test(num_demands):
    """Runs the test for a given number of demands and returns results."""
    print(f"--- Running test for {num_demands} demand points ---")
    start_time = time.time()
    
    try:
        process = subprocess.run(
            ['python', 'test_debugged_precise.py', '--num_demands', str(num_demands)],
            capture_output=True,
            text=True,
            timeout=300  # 5-minute timeout per run
        )
        
        end_time = time.time()
        solve_time = end_time - start_time
        
        if process.returncode == 0 and "SUCCESS!" in process.stdout:
            print(f"  SUCCESS in {solve_time:.2f} seconds.")
            return True, solve_time
        else:
            print(f"  FAILURE in {solve_time:.2f} seconds.")
            print("  --- STDOUT ---")
            print(process.stdout)
            print("  --- STDERR ---")
            print(process.stderr)
            return False, solve_time
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        solve_time = end_time - start_time
        print(f"  FAILURE: Timed out after {solve_time:.2f} seconds.")
        return False, solve_time

def main():
    """Main function to run the robustness tests."""
    print("============================================================")
    print("ROBUSTNESS TEST FOR PRECISE METHOD")
    print("============================================================")
    
    demand_points_to_test = [5, 10, 15, 20]
    results = {}
    
    for num_demands in demand_points_to_test:
        success, solve_time = run_test(num_demands)
        results[num_demands] = {'success': success, 'time': solve_time}
    
    print("\n============================================================")
    print("ROBUSTNESS TEST SUMMARY")
    print("============================================================")
    
    for num_demands, result in results.items():
        status = "Success" if result['success'] else "Failure"
        print(f"  {num_demands} demands: {status} in {result['time']:.2f} seconds")
        
if __name__ == "__main__":
    main() 