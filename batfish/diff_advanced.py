import argparse
import os
import sys
from pybatfish.client.session import Session

def get_node_set(traces):
    """Extracts a unique set of all router names visited in a list of traces."""
    nodes = set()
    for trace in traces:
        for hop in trace.hops:
            nodes.add(hop.node)
    return nodes

def main():
    parser = argparse.ArgumentParser(description='General Waypointing and Load Balancing Discovery.')
    parser.add_argument('--folder', required=True, help='Experiment folder')
    parser.add_argument('--c1', required=True, help='Candidate 1 folder')
    parser.add_argument('--c2', required=True, help='Candidate 2 folder')
    args = parser.parse_args()

    # Batfish Setup
    bf = Session(host="localhost")
    bf.set_network(f"gen_analysis_{os.path.basename(args.folder)}")
    
    path_c1 = os.path.join(args.folder, args.c1)
    path_c2 = os.path.join(args.folder, args.c2)

    print(f"--- Initializing Snapshots ---")
    bf.init_snapshot(path_c1, name="snap1", overwrite=True)
    bf.init_snapshot(path_c2, name="snap2", overwrite=True)

    # Perform Reachability
    # We use a broad reachability check to find ANY common successful flow
    print("Searching for behavioral divergences...")
    reach_1 = bf.q.reachability().answer(snapshot="snap1").frame()
    reach_2 = bf.q.reachability().answer(snapshot="snap2").frame()

    if reach_1.empty or reach_2.empty:
        print("No successful flows found to compare.")
        return

    # To be precise, we find a flow that exists in both results
    # We'll compare the first one found in Candidate 1 that also exists in Candidate 2
    for _, row1 in reach_1.iterrows():
        flow = row1['Flow']
        match_c2 = reach_2[reach_2['Flow'] == flow]
        
        if not match_c2.empty:
            row2 = match_c2.iloc[0]
            
            # 1. Compare Load Balancing (Number of Paths)
            paths_c1 = len(row1['Traces'])
            paths_c2 = len(row2['Traces'])
            
            # 2. Compare Waypointing (Sets of Nodes)
            nodes_c1 = get_node_set(row1['Traces'])
            nodes_c2 = get_node_set(row2['Traces'])
            
            diff_lb = paths_c1 != paths_c2
            diff_wp = nodes_c1 != nodes_c2

            if diff_lb or diff_wp:
                print(f"\n[DIVERGENCE DISCOVERED]")
                print(f"Flow: {flow}")
                
                if diff_wp:
                    only_in_c1 = nodes_c1 - nodes_c2
                    only_in_c2 = nodes_c2 - nodes_c1
                    print(f"- Waypointing Difference:")
                    if only_in_c1: print(f"  * Candidate 1 uniquely transits: {only_in_c1}")
                    if only_in_c2: print(f"  * Candidate 2 uniquely transits: {only_in_c2}")
                
                if diff_lb:
                    print(f"- Load Balancing Difference:")
                    print(f"  * Candidate 1 has {paths_c1} paths, Candidate 2 has {paths_c2} paths.")
                
                # We found a representative difference, so we can stop searching
                break
    else:
        print("All successful flows behave identically in both candidates.")

if __name__ == "__main__":
    main()