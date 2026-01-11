"""
Parallel Training Script for NashCFRAgent.
This script distributes the training of CFR policies across multiple CPU cores.
"""
import os
import sys
import argparse
import glob
import pickle
import multiprocessing
import time
from itertools import product
from torch.utils.tensorboard import SummaryWriter

# Ensure we can import the package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from liars_dice.agents.nash_agent import NashCFRAgent

def train_worker(args):
    """
    Worker function executed by each process.
    pops a configuration, trains it, and saves the partial result.
    """
    dice_counts, faces, iterations, weights_dir, tensorboard_logdir, worker_id = args
    
    # Construct a unique string for this configuration key
    dice_str = "_".join(map(str, dice_counts))
    filename = f"policy_part_{dice_str}.pkl"
    filepath = os.path.join(weights_dir, filename)
    
    # TensorBoard setup if logdir is provided
    tb_writer = None
    if tensorboard_logdir:
        # Use a subdirectory or unique suffix to separate workers if writing to same root
        # Here we just use the main logdir, standard TensorBoard usage supports multiple event files
        # But adding a subfolder per config is cleaner for "runs"
        log_path = os.path.join(tensorboard_logdir, f"{dice_str}")
        tb_writer = SummaryWriter(log_dir=log_path)

    print(f"[Worker {worker_id}] Started training config: {dice_counts}", flush=True)
    start_time = time.time()
    
    # Callback to log progress
    def progress_monitor(iteration, delta):
        if tb_writer:
            tb_writer.add_scalar(f"convergence_delta/max_delta", delta, iteration)
            tb_writer.flush()
    
    # Generate a unique seed based on config to ensure reproducibility but independence
    # We use a hash of the config + a base seed
    seed = (42 + hash(dice_counts)) % 2**32
    
    # Execute Training Logic
    # We call the static method directly to avoid instantiating the full agent
    policy, metrics = NashCFRAgent.train_cfr_policy(
        dice_counts=dice_counts,
        faces=faces,
        iterations=iterations,
        seed=seed,
        track_regret=True, # Enable convergence tracking
        check_convergence_every=max(100, iterations // 100),
        progress_callback=progress_monitor
    )
    
    converged = metrics.get('converged', False)
    status_str = "Converged" if converged else "Max Iterations"
    duration = time.time() - start_time
    
    print(f"[Worker {worker_id}] Finished {dice_counts} in {duration:.2f}s - {status_str}", flush=True)

    if tb_writer:
        # Log final status as text
        tb_writer.add_text("status", status_str, iterations)
        # Log boolean as scalar for easy filtering (1=Converged, 0=Not)
        tb_writer.add_scalar("convergence_status", 1 if converged else 0, iterations)
        tb_writer.close()
    
    # Prepare data for saving
    # The key format matches what the agent expects: (dice_counts, faces_tuple)
    key = (dice_counts, tuple(faces))
    save_data = {
        'policies': {key: policy}, 
        'metrics': {key: metrics}
    }
    
    # Save partial file (overwrite if exists)
    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)
        
    return filepath

def merge_partial_policies(weights_dir, output_path):
    """
    Merges all policy_part_*.pkl files in the weights directory into a single master file.
    """
    print(f"\n--- Starting Merge Process in {weights_dir} ---")
    files = glob.glob(os.path.join(weights_dir, "policy_part_*.pkl"))
    
    if not files:
        print("No partial policy files found to merge.")
        return

    master_policies = {}
    master_metrics = {}
    
    for f_path in files:
        try:
            with open(f_path, "rb") as f:
                data = pickle.load(f)
                
                # Handle both raw dict and wrapped dict formats
                if isinstance(data, dict) and 'policies' in data:
                    p_data = data['policies']
                    m_data = data.get('metrics', {})
                else:
                    p_data = data
                    m_data = {}
                
                master_policies.update(p_data)
                master_metrics.update(m_data)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    # Save the final merged file
    final_data = {'policies': master_policies, 'metrics': master_metrics}
    NashCFRAgent.save_policy_dict(final_data, output_path)
    print(f"Successfully merged {len(master_policies)} configurations into {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-process CFR Training for Liar's Dice")
    parser.add_argument('--num_players', type=int, default=2, help='Number of players in the game')
    parser.add_argument('--max_dice', type=int, default=5, help='Maximum dice per player to train up to')
    parser.add_argument('--faces', type=int, nargs='+', default=[1,2,3,4,5,6], help='Faces of the dice (e.g. 1 2 3 4 5 6)')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of CFR iterations per configuration')
    parser.add_argument('--cores', type=int, default=None, help='Number of processes to accept (default: all available CPUs)')
    parser.add_argument('--output', type=str, default=None, help='Path for the final merged policy file')
    parser.add_argument('--tensorboard', type=str, default=None, help='Directory for TensorBoard logs')
    
    args = parser.parse_args()
    
    # Configuration: Robust Core Detection (Slurm-aware)
    if args.cores:
        num_cores = args.cores
    elif "SLURM_CPUS_PER_TASK" in os.environ:
        # Slurm sets this variable to the number of CPUs requested per task
        num_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
        print(f"Detected Slurm environment. Using {num_cores} cores allocated.")
    else:
        # Fallback: Try to respect affinity (containers) or use all CPUs
        if hasattr(os, "sched_getaffinity"):
            try:
                num_cores = len(os.sched_getaffinity(0))
            except Exception:
                num_cores = os.cpu_count()
        else:
            num_cores = os.cpu_count()

    faces = tuple(args.faces)
    
    # Path setup
    # Determine weights directory relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, '..', 'liars_dice', 'agents', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    output_path = args.output if args.output else os.path.join(weights_dir, "nash_cfr_policy.pkl")
    
    # Generate all dice combinations to train
    # product(range(1, 6), range(1, 6)) for 2 players gives (1,1), (1,2)... (5,5)
    dice_ranges = [range(1, args.max_dice + 1) for _ in range(args.num_players)]
    all_configs = list(product(*dice_ranges))
    
    print(f"--- Initialization ---")
    print(f"Target: {len(all_configs)} configurations")
    print(f"Workers: {num_cores}")
    print(f"Iterations: {args.iterations}")
    print(f"Output Dir: {weights_dir}")
    print(f"----------------------\n")
    
    # Prepare arguments for each worker
    # We add an index to help with logging/debugging if needed
    worker_inputs = []
    for i, config in enumerate(all_configs):
        # inputs: (dice_counts, faces, iterations, weights_dir, tensorboard_logdir, worker_id)
        worker_inputs.append((config, faces, args.iterations, weights_dir, args.tensorboard, i % num_cores))
    
    # Run multiprocessing pool
    # The Pool will automatically manage the queue of tasks (worker_inputs)
    # and distribute them to the available worker processes.
    try:
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(train_worker, worker_inputs)
    except KeyboardInterrupt:
        print("\nStopping training prematurely...")
        pool.terminate()
        pool.join()
        sys.exit(1)
        
    print("\nAll workers completed.")
    
    # Merge results
    merge_partial_policies(weights_dir, output_path)

if __name__ == "__main__":
    # Essential for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
