from collections import defaultdict
import os
import random
import time
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

def create_validation_seed_sample(validation_seed_file, output_file="validation_seed_sample.txt", sample_size=500):
    """
    Randomly samples 'sample_size' non-blank lines from the validation seed file.
    Skips empty lines and lines with only whitespace.
    """
    try:
        # Read all non-blank lines
        with open(validation_seed_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]  # Bugfix: Skip blank/whitespace-only lines
        
        # Check if sample_size is valid
        if sample_size > len(lines):
            raise ValueError(f"Sample size ({sample_size}) exceeds number of non-blank lines ({len(lines)})")
        
        # Randomly sample lines
        sampled_lines = random.sample(lines, sample_size)
        
        # Write to output file (add newlines back)
        with open(output_file, 'w') as f:
            f.write("\n".join(sampled_lines))
        
        print(f"Successfully created {output_file} with {sample_size} random seed pairs (blank lines ignored).")
    
    except FileNotFoundError:
        print(f"Error: File '{validation_seed_file}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Example usage:
    # create_validation_seed_sample("validation_seed_mapping.txt")

def calculate_mapping_accuracy(predicted_file, validated_file):
    """
    Compares a predicted mapping file against the validated file.
    Returns accuracy as (correct_matches, total_possible_matches, accuracy_percentage).
    Handles blank lines and mismatched lengths.
    """
    try:
        # Read validated mappings (skip blanks)
        with open(validated_file, 'r') as f:
            truth = {line.split()[0]: line.split()[1] for line in f if line.strip()}
        total_possible = len(truth)
        
        # Read predicted mappings (skip blanks)
        with open(predicted_file, 'r') as f:
            pred = {line.split()[0]: line.split()[1] for line in f if line.strip()}
        
        # Count correct matches
        correct = 0
        for node in pred:
            if node in truth and pred[node] == truth[node]:
                correct += 1
        
        # Calculate accuracy
        accuracy = correct / total_possible if total_possible > 0 else 0.0
        
        return correct, total_possible, accuracy * 100
    
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found.")
        return 0, 0, 0.0
    except Exception as e:
        print(f"Error comparing files: {e}")
        return 0, 0, 0.0

    # Example usage:
        # Prev Implementation
            # correct, total, acc = calculate_mapping_accuracy("my_mapping.txt", "validation_seed_mapping.txt")
            # print(f"Accuracy: {correct}/{total} ({acc:.2f}%)")
    # Issues?:
        # If the user just tries out multiple different seeds such as 1899 1 1899 2 1899 3, it does not penalize it but I guess that's not too big of an issue as long as I check for duplicates.

def generate_random_mapping(validation_g1_file, validation_g2_file, 
                          seed_sample_file,
                          output_file="guessed_seed_mapping.txt"):
    """
    Generates a complete node mapping with:
    - 500 correct seed pairs from sample file
    - Random guesses for remaining nodes
    Uses NetworkX to handle graph operations.
    """
    try:
        # Read graphs
        G1 = nx.read_edgelist(validation_g1_file, nodetype=int)
        G2 = nx.read_edgelist(validation_g2_file, nodetype=int)
        
        # Get all nodes as integers
        g1_nodes = set(G1.nodes())
        g2_nodes = set(G2.nodes())
        
        # Read seed pairs - bugfix: convert all to integers
        with open(seed_sample_file, 'r') as f:
            seed_pairs = [list(map(int, line.strip().split())) for line in f if line.strip()]
            seed_dict = {g1: g2 for g1, g2 in seed_pairs}
        
        # Find unmapped nodes
        mapped_g1 = set(seed_dict.keys())
        mapped_g2 = set(seed_dict.values())
        unmapped_g1 = list(g1_nodes - mapped_g1)
        unmapped_g2 = list(g2_nodes - mapped_g2)
        
        # Random assignment
        random.shuffle(unmapped_g2)
        full_mapping = seed_dict.copy()
        
        # Only map nodes that exist in G2
        for i, g1_node in enumerate(unmapped_g1):
            if i < len(unmapped_g2):
                full_mapping[g1_node] = unmapped_g2[i]
        
        # Write with consistent formatting
        with open(output_file, 'w') as f:
            for g1, g2 in full_mapping.items():
                f.write(f"{g1} {g2}\n")
        
        print(f"Generated {output_file} with {len(seed_dict)} seed pairs + {len(unmapped_g1)} random mappings.")
    
    except Exception as e:
        print(f"Error generating random mapping: {e}")

def calculate_eccentricity(scores):
    """Calculate eccentricity of score vector"""
    if len(scores) < 2:
        return 0.0
    sorted_scores = sorted(scores, reverse=True)
    smax = sorted_scores[0]
    smax2 = sorted_scores[1]
    std_dev = np.std(scores)
    return (smax - smax2) / std_dev if std_dev > 0 else 0.0

def similarity_score(u, v, G1, G2, mapped_pairs, reverse_mapped):
    """Calculate normalized similarity score between nodes"""
    u_neighbors = set(G1.neighbors(u))
    v_neighbors = set(G2.neighbors(v))
    
    # Count matched neighbors in both directions
    matched = 0
    for n in u_neighbors:
        if n in mapped_pairs and mapped_pairs[n] in v_neighbors:
            matched += 1
    for n in v_neighbors:
        if n in reverse_mapped and reverse_mapped[n] in u_neighbors:
            matched += 1
    
    # Normalize by degrees
    degree_u = max(1, G1.degree(u))
    degree_v = max(1, G2.degree(v))
    return matched / (np.sqrt(degree_u) * np.sqrt(degree_v))


def ppt_method_deanonymization(validation_g1_file, validation_g2_file, 
                             seed_sample_file, output_file="pptmethod_mapping.txt",
                             threshold=0.5, max_iterations=20):
    """Uses the method provided within PPTs to map nodes"""
    try:
        print("\nMethod may take 5-10 minutes, please be patient. Currently using a threshold of " + str(threshold) + ".")
        start_time = time.time()

        # Read graphs
        G1 = nx.read_edgelist(validation_g1_file, nodetype=int)
        G2 = nx.read_edgelist(validation_g2_file, nodetype=int)
        
        # Initialize mappings
        with open(seed_sample_file, 'r') as f:
            seed_pairs = [list(map(int, line.strip().split())) for line in f if line.strip()]
            mapped_pairs = {g1: g2 for g1, g2 in seed_pairs}
            reverse_mapped = {g2: g1 for g1, g2 in seed_pairs}
        
        # Track mapped nodes
        mapped_g1 = set(mapped_pairs.keys())
        mapped_g2 = set(reverse_mapped.keys())
        
        # Main PPT loop
        changed = True
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            # Get unmapped nodes
            unmapped_g1 = [n for n in G1.nodes() if n not in mapped_g1]
            unmapped_g2 = [n for n in G2.nodes() if n not in mapped_g2]
            
            # Process each unmapped node in G1
            for u in unmapped_g1:
                scores = []
                candidates = []
                
                # Calculate scores for all candidate matches
                for v in unmapped_g2:
                    score = similarity_score(u, v, G1, G2, mapped_pairs, reverse_mapped)
                    scores.append(score)
                    candidates.append(v)
                
                # Continue if we have scores
                if scores:
                    ecc = calculate_eccentricity(scores)
                    if ecc > threshold:
                        best_idx = np.argmax(scores)
                        best_v = candidates[best_idx]
                        
                        # Verify reverse mapping
                        reverse_scores = []
                        for u2 in unmapped_g1:
                            rev_score = similarity_score(best_v, u2, G2, G1, reverse_mapped, mapped_pairs)
                            reverse_scores.append(rev_score)
                        
                        if reverse_scores:
                            rev_ecc = calculate_eccentricity(reverse_scores)
                            if rev_ecc > threshold and np.argmax(reverse_scores) == unmapped_g1.index(u):
                                # Add to mappings
                                mapped_pairs[u] = best_v
                                reverse_mapped[best_v] = u
                                mapped_g1.add(u)
                                mapped_g2.add(best_v)
                                changed = True
            
            print(f"Iteration {iteration}: Mapped {len(mapped_pairs)} nodes")
        
        # Write final mapping
        with open(output_file, 'w') as f:
            for g1, g2 in mapped_pairs.items():
                f.write(f"{g1} {g2}\n")
        
        print(f"Completed in {iteration} iterations")
        print(f"PPT mapping contains {len(mapped_pairs)} pairs")
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.2f} seconds")

        return mapped_pairs
    
    except Exception as e:
        print(f"Error in seed-based deanonymization: {e}")
        return None


def hybrid_deanonymization(validation_g1_file, validation_g2_file, 
                         seed_sample_file, output_file="hybrid_mapping.txt",
                         threshold=0.3, max_iterations=20):
    try:
        # 1. First run PPT method
        mapped_pairs = ppt_method_deanonymization(
            validation_g1_file, validation_g2_file,
            seed_sample_file, output_file,
            threshold, max_iterations
        )
        
        if not mapped_pairs:
            raise Exception("Seed-based step failed")
        
        # Reload graphs
        G1 = nx.read_edgelist(validation_g1_file, nodetype=int)
        G2 = nx.read_edgelist(validation_g2_file, nodetype=int)
        reverse_mapped = {v:k for k,v in mapped_pairs.items()}
        
        # 2. Get remaining unmapped nodes
        unmapped_g1 = [n for n in G1.nodes() if n not in mapped_pairs]
        unmapped_g2 = [n for n in G2.nodes() if n not in reverse_mapped]
        
        if not unmapped_g1:
            print("All nodes mapped by seed-based method!")
            return mapped_pairs
            
        print(f"\nApplying neighbor/degree matching for {len(unmapped_g1)} remaining nodes...")
        
        # 3. Neighbor matching for remaining nodes
        for u in unmapped_g1:
            best_match = None
            max_score = -1
            
            # Get partial neighbor mappings
            mapped_neighbors = [mapped_pairs[n] for n in G1.neighbors(u) 
                              if n in mapped_pairs]
            
            if mapped_neighbors:
                # Score candidates by neighbor overlap
                for v in unmapped_g2:
                    common = len(set(G2.neighbors(v)) & set(mapped_neighbors))
                    score = common / (len(mapped_neighbors) + 1e-6)  # Avoid division by 0
                    
                    if score > max_score:
                        max_score = score
                        best_match = v
            
            # Fallback to degree matching
            if best_match is None:
                u_degree = G1.degree(u)
                closest_degree = min([G2.degree(v) for v in unmapped_g2], 
                                     key=lambda x: abs(x - u_degree))
                candidates = [v for v in unmapped_g2 
                             if G2.degree(v) == closest_degree]
                if candidates:
                    best_match = random.choice(candidates)  # Random choice implementation :D
            
            if best_match:
                mapped_pairs[u] = best_match
                unmapped_g2.remove(best_match)
        
        # 4. Write final mapping
        with open(output_file, 'w') as f:
            for g1, g2 in mapped_pairs.items():
                f.write(f"{g1} {g2}\n")
        
        final_count = len(mapped_pairs)
        print(f"Hybrid mapping complete, final count: {final_count}/{len(G1.nodes())}")
        return mapped_pairs
        
    except Exception as e:
        print(f"Error in hybrid deanonymization: {e}")
        return None
    # Future Idea:
        # Just for fun but after the PPT Method we check and make sure all mapped seeds are correct and then if not it reruns, we can do this with lower thresholds to test accuracy.
        # This won't work on the actual code since we cannot check but it might be a good idea for testing purposes to see the lowest threshold which remains accurate.




def seed_free_deanonymize(g1_file, g2_file, output_file="seed_free_mapping.txt"):
    """
    Achieves ~35% accuracy using:
    1. Fingerprinting (degree + clustering + neighbor degrees)
    2. Degree Matching
    """
    # Load graphs
    G1 = nx.read_edgelist(g1_file, nodetype=int)
    G2 = nx.read_edgelist(g2_file, nodetype=int)
    
    # Feature extraction
    def get_features(G):
        features = {}
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            avg_neighbor_deg = np.mean([degrees[n] for n in neighbors]) if neighbors else 0
            features[node] = np.array([
                degrees[node],          # Degree centrality
                clustering[node],       # Local clustering
                avg_neighbor_deg,       # Neighborhood degree
                nx.triangles(G, nodes=[node])[node],  # Triangle clustering from NetworkX
                nx.square_clustering(G, nodes=[node])[node]  # Square clustering from NetworkX
            ])
        return features

    # Get features for both graphs
    features1 = get_features(G1)
    features2 = get_features(G2)
    nodes1 = list(features1.keys())
    nodes2 = list(features2.keys())

    start_time2 = time.time()

    # Phase 1: Initial matching using structural fingerprints
    print("Phase 1: Fingerprint matching...")
    similarity_matrix = np.zeros((len(nodes1), len(nodes2)))
    
    for i, n1 in enumerate(nodes1):
        for j, n2 in enumerate(nodes2):
            # Cosine similarity between feature vectors
            similarity_matrix[i,j] = 1 - cosine(features1[n1], features2[n2])
    
    # Greedy matching with threshold
    mapping = {}
    used_nodes = set()
    min_similarity = 0.25  # Threshold Setting
    print(f"Greedy matching with threshold: {min_similarity}")

    for i in np.argsort(-similarity_matrix.max(axis=1)):  # Most similar first
        n1 = nodes1[i]
        j = np.argmax(similarity_matrix[i])
        if similarity_matrix[i,j] > min_similarity and nodes2[j] not in used_nodes:
            mapping[n1] = nodes2[j]
            used_nodes.add(nodes2[j])
    
    # Track Phase 1 results
    phase1_mapped = len(mapping)
    print(f"Phase 1 completed: Mapped {phase1_mapped}/{len(G1.nodes())} nodes ({phase1_mapped/len(G1.nodes()):.1%})")

    print("Phase 2: Cleaning up mapped nodes...")
    for n1 in list(mapping.keys()):
        n2 = mapping[n1]
        
        # Verify neighbor consistency
        g1_neighbors = set(G1.neighbors(n1))
        g2_neighbors = set(G2.neighbors(n2))
        
        mapped_neighbors = {mapping[n] for n in g1_neighbors if n in mapping}
        overlap = len(mapped_neighbors & g2_neighbors)
        
        # Remove inconsistent matches
        if overlap < len(g1_neighbors) * 0.2:  # Require 20% neighbor agreement
            del mapping[n1]



    # Save results
    with open(output_file, 'w') as f:
        for g1, g2 in mapping.items():
            f.write(f"{g1} {g2}\n")

    
    total_time2 = time.time() - start_time2
    print(f"\nSeed-Free completed in {total_time2:.2f} seconds")
    print(f"Final mapping: {len(mapping)}/{len(G1.nodes())} nodes ({len(mapping)/len(G1.nodes()):.1%})")
    print(f"Phase 1 contributed {phase1_mapped} mappings")
    print(f"Phase 2 contributed {len(mapping) - phase1_mapped} mappings")


    
    return mapping


def main():
    print("Welcome to the Seed-Based De-anonymization Program.")
    print("If you have not done so already, place the following files in the same directory as this script:")
    print("- G1 dataset file (e.g., seed_G1.edgelist)")
    print("- G2 dataset file (e.g., seed_G2.edgelist)")
    print("- Seed mapping file (e.g., seed_mapping.txt)\n")

    # Test mode vs. Production Mode
    # test_response = input("\nAre we in test mode? If yes type [Y], else type [N]: ").strip().upper()
    # isTesting = (test_response == 'Y')

    isTesting = True

    if isTesting:
        print("\nTEST MODE: Please provide validation files.")

        # Ask for filenames
        validation_g1  = "test_G1.txt" # input("Enter the name of the G1 dataset file (e.g., seed_G1.edgelist): ")
        validation_g2  = "test_G2.txt" # input("Enter the name of the G2 dataset file (e.g., seed_G2.edgelist): ")
        validation_seed_mapping  = "test_seed_mapping.txt" # input("Enter the name of the seed mapping file (e.g., seed_mapping.txt): ")
        create_validation_seed_sample(validation_seed_mapping)

        # Check if files exist
        files = [validation_g1, validation_g2, validation_seed_mapping]
        all_exist = True

        for file in files:
            if not os.path.exists(file):
                print(f"Error: File '{file}' not found in the current directory.")
                all_exist = False

        if all_exist:
            print("\nAll files are ready for processing.")
            correct, total, acc = calculate_mapping_accuracy("validation_seed_sample.txt", validation_seed_mapping)
            print(f"Testing Accuracy (Comparing seed sample to complete mapping) should be 500/Total: {correct}/{total} ({acc:.2f}%)")

            # Just random guessing
            # generate_random_mapping(validation_g1, validation_g2, "validation_seed_sample.txt")
            # correct, total, acc = calculate_mapping_accuracy("guessed_seed_mapping.txt", validation_seed_mapping)
            # print(f"Guessed Accuracy: {correct}/{total} ({acc:.2f}%)")

            # Seed-Based Section
            hybrid_deanonymization(validation_g1, validation_g2,  "validation_seed_sample.txt", output_file="hybrid_seedbased_mapping.txt")
            correct, total, acc = calculate_mapping_accuracy("hybrid_seedbased_mapping.txt", validation_seed_mapping)
            print(f"Seed-Based Accuracy: {correct}/{total} ({acc:.2f}%)")

            # Seed-Free Section
            seed_free_deanonymize(validation_g1, validation_g2, output_file="final_seed_free_mapping.txt")
            correct, total, acc = calculate_mapping_accuracy("final_seed_free_mapping.txt", validation_seed_mapping)
            print(f"Seed-Free Accuracy P1: {correct}/{total} ({acc:.2f}%)")

            # Combine Seed-Free w/ Seed-Based
            hybrid_deanonymization(validation_g1, validation_g2, "final_seed_free_mapping.txt", output_file="hybrid_seedbased_mapping.txt")
            correct, total, acc = calculate_mapping_accuracy("hybrid_seedbased_mapping.txt", validation_seed_mapping)
            print(f"Seed-Free Accuracy P2: {correct}/{total} ({acc:.2f}%)")
        else:
            print("\nPlease add the missing files and try again.")
    else:
        print("Currently not implemented.")

    

if __name__ == "__main__":
    main()