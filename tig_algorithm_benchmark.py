import json
import os
import sys
import csv
import argparse
import subprocess
from datetime import datetime
import requests
import random
    
def set_checkout(repo_name, branch_name):
    try:
        subprocess.check_output(
            ['git', 'checkout', branch_name],
            stderr=subprocess.STDOUT,
            cwd=repo_name
        )
        
    except subprocess.CalledProcessError as e:
        print(f"Error on checkout to branch '{branch_name}': ", e.output.decode('utf-8'))

    try:
        subprocess.check_output(
            ['git', 'pull'],
            stderr=subprocess.STDOUT,
            cwd=repo_name
        )

    except subprocess.CalledProcessError as e:
        print("Error on pull from branch '{branch_name}': ", e.output.decode('utf-8'))
        
    return True

def get_closest_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p

def main():
    parser = argparse.ArgumentParser(description="Benchmark TIG Worker algorithms")
    parser.add_argument("--log_file", default="tig_dataset_algorithms_benchmark.csv", help="Path to the log file")    
    parser.add_argument("--vm_flavour", default="n1-cpu-small", help="VM flavour (default: 'n1-cpu-small')")
    parser.add_argument("--num_gpus", default=0, type=int, help="Number of GPUs (default: 0)")
    parser.add_argument("--num_cpus", default=16, type=int, help="Number of CPUs (default: 4)")
    parser.add_argument("--start_nonce", default=1, type=int, help="Start nonce (default: 0)")
    parser.add_argument("--num_nonces", default=23040, type=int, help="Number of nonces")
    parser.add_argument("--challenges_name", default="[vehicle_routing/advanced_routing, knapsack/classic_quadkp, satisfiability/sat_global_opt, vector_search/invector_hybrid]", help="Challenges name (default: '[vector_search/invector_hybrid]')")
    parser.add_argument("--max_minutes_per_algo", default=4, type=int, help="Max minutes per algorithm (default: 4)")

    args = parser.parse_args()
    
    REPO_TIG_MONOREPO = os.path.join("..", "tig-monorepo")
    tig_worker_path = os.path.join(REPO_TIG_MONOREPO, "target/release/tig-worker")
    
    if not os.path.isfile(tig_worker_path):
        sys.exit(f"Error: tig-worker binary '{tig_worker_path}' not found.")
    print(f"Found tig-worker binary at {tig_worker_path}")
        
    if not os.path.isfile(args.log_file):
        with open(args.log_file, mode='w') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                "Timestamp", "VM_Flavour", "Num_GPUs", "Num_CPUs", "Branch_Name", "Challenge_ID", 
                "Challenge_Name", "Difficulty", "Algorithm_ID", "Start_Time", 
                "End_Time", "Duration", "Nonce", "Solutions_Count", 
                "Invalid_Count", "Output_Stdout", "Output_Stderr"
            ])
    
    challenge_ids = {
        "satisfiability": "c001",
        "vehicle_routing": "c002",
        "knapsack": "c003",
        "vector_search": "c004"
    }

    challenge_name_list = args.challenges_name.strip('[]').replace(" ", "").split(',')

    response = requests.get("https://mainnet-api.tig.foundation/get-block")
    if response.status_code != 200:
        sys.exit(f"Error: Failed to get block data from API. Status code: {response.status_code}")
    block_data = response.json()
    block_id = block_data["block"]["id"]

    response = requests.get(f"https://mainnet-api.tig.foundation/get-challenges?block_id={block_id}")
    if response.status_code != 200:
        sys.exit(f"Error: Failed to get challenges data from API. Status code: {response.status_code}")
    challenges_data = response.json()

    for branch_name in challenge_name_list:
    
        remaining_nonces = args.num_nonces
        current_nonce = args.start_nonce

        challenge, algorithm = str.split(branch_name, "/")
        challenge_id = challenge_ids.get(challenge)
        if not challenge_id:
            sys.exit(f"Error: Challenge '{challenge}' not found in the challenge_ids dictionary.")

        if not set_checkout(REPO_TIG_MONOREPO, branch_name):
            sys.exit(f"Error: Failed to checkout to branch '{branch_name}'.")

        challenge_data = next((c for c in challenges_data["challenges"] if c["id"] == challenge_id), None)
        if not challenge_data:
            sys.exit(f"Error: Challenge '{challenge}' not found in the API response.")
        difficults = challenge_data["block_data"]["qualifier_difficulties"]

        wasm_file = os.path.join(REPO_TIG_MONOREPO, "tig-algorithms/wasm", challenge, f"{algorithm}.wasm")
        if not os.path.isfile(wasm_file):
            print(f"Error: Algorithm '{algorithm}.wasm' not found in the repository.")
            return
                
        print(f"Testing algorithm: {algorithm} for challenge: {challenge}")

        algo_start_time = int(datetime.now().timestamp() * 1000)
        while remaining_nonces > 0 and (int(datetime.now().timestamp() * 1000) - algo_start_time) < (60 * 1000 * args.max_minutes_per_algo):
            difficulty = random.choice(difficults)   

            nonces_to_compute = min(args.num_cpus, remaining_nonces)
            power_of_2 = get_closest_power_of_2(nonces_to_compute)
            start_time = int(datetime.now().timestamp() * 1000)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            settings = json.dumps({
                "challenge_id": challenge_id,
                "difficulty": difficulty,
                "algorithm_id": algorithm,
                "player_id":"",
                "block_id":""
            })

            command = [
                tig_worker_path, "compute_batch", 
                settings, "random_string", str(current_nonce), str(nonces_to_compute),
                str(power_of_2), wasm_file,
                "--workers", str(args.num_cpus)
            ]

            result = subprocess.run(
                command,
                text=True,
                capture_output=True
            )

            output_stdout = result.stdout
            output_stderr = result.stderr
            
            end_time = int(datetime.now().timestamp() * 1000)
            duration = end_time - start_time
            
            output_stdout = json.loads(output_stdout)
            solutions_count = len(output_stdout.get("solution_nonces", []))
            invalid_count = nonces_to_compute - solutions_count

            with open(args.log_file, 'a') as log:
                writer = csv.writer(log, delimiter=';')
                writer.writerow([
                    timestamp, args.vm_flavour, args.num_gpus, args.num_cpus, branch_name, challenge_id,
                    challenge, difficulty, algorithm, start_time, end_time, 
                    duration, current_nonce, solutions_count, invalid_count, output_stdout, 
                    output_stderr
                ])
            
            current_nonce += nonces_to_compute
            remaining_nonces -= nonces_to_compute

    print(f"Testing complete for all algorithms in challenge '{challenge}'. Results logged to {args.log_file}.")

if __name__ == "__main__":
    main()
