import argparse
import os
import sys
from pybatfish.client.session import Session

def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description='Run Batfish differential analysis on two candidates.')
    parser.add_argument('--folder', required=True, help='The experiment folder (e.g., bgp_simple)')
    parser.add_argument('--c1', required=True, help='Name of the first candidate folder (e.g., candidate1)')
    parser.add_argument('--c2', required=True, help='Name of the second candidate folder (e.g., candidate2)')
    parser.add_argument('--host', default='localhost', help='Batfish server address (default: localhost)')

    args = parser.parse_args()

    # 2. Construct Paths
    # Assumes structure: folder/c1/configs/ and folder/c2/configs/
    path_c1 = os.path.join(args.folder, args.c1)
    path_c2 = os.path.join(args.folder, args.c2)

    # Validate paths exist
    for p in [path_c1, path_c2]:
        if not os.path.exists(p):
            print(f"Error: Path not found: {p}")
            sys.exit(1)

    # 3. Initialize Batfish Session
    bf = Session(host=args.host)
    network_name = f"analysis_{os.path.basename(args.folder)}"
    bf.set_network(network_name)

    print(f"--- Analyzing {args.folder}: {args.c1} vs {args.c2} ---")

    try:
        # Initialize Snapshots
        bf.init_snapshot(path_c1, name="snap_1", overwrite=True)
        bf.init_snapshot(path_c2, name="snap_2", overwrite=True)

        # 4. Perform Differential Reachability Analysis
        # Finds a flow that is PERMITTED in snap_1 but DENIED in snap_2
        results = bf.q.differentialReachability().answer(
            snapshot="snap_1", 
            reference_snapshot="snap_2"
        ).frame()

        # 5. Output the Difference
        if not results.empty:
            print("\n[DIFFERENCE DETECTED]")
            # Get the first counter-example
            flow = results.iloc[0]['Flow']
            print(f"Counter-example Packet (Packet P): {flow}")
            
            print("\nTrace in Candidate 1 (Permitted):")
            print(results.iloc[0]['Snapshot_Traces'])
            
            print("\nTrace in Candidate 2 (Denied/Different):")
            print(results.iloc[0]['Reference_Traces'])
        else:
            print("\n[NO DIFFERENCE FOUND]")
            print("The two configurations are behaviorally identical for reachability.")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()