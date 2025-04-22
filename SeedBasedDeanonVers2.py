import os
import random
import networkx as nx
import numpy as np
from collections import defaultdict


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

def seed_based_deanonymization(validation_g1_file, validation_g2_file, 
                             seed_sample_file, output_file="pptmethod_mapping.txt",
                             threshold=0.5, max_iterations=20):
    """Uses the method provided within PPTs to map nodes"""
    try:
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
        
        # Main propagation loop
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
        print(f"Final mapping contains {len(mapped_pairs)} pairs")
        return mapped_pairs
    
    except Exception as e:
        print(f"Error in seed-based deanonymization: {e}")
        return None


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
                # calculate_mapping_accuracy("validation_seed_sample.txt", validation_seed_mapping)
            correct, total, acc = calculate_mapping_accuracy("validation_seed_sample.txt", validation_seed_mapping)
            print(f"Accuracy: {correct}/{total} ({acc:.2f}%)")

            seed_based_deanonymization(validation_g1, validation_g2, "validation_seed_sample.txt")
            correct, total, acc = calculate_mapping_accuracy("pptmethod_mapping.txt", validation_seed_mapping)
            print(f"PPT Method Accuracy: {correct}/{total} ({acc:.2f}%)")

        else:
            print("\nPlease add the missing files and try again.")
    else:
        print("Currently not implemented.")

    

if __name__ == "__main__":
    main()