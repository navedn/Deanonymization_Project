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
        
        print(f"Successfully created {output_file} with {sample_size} random seed pairs.")
    
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

def ppt_method_deanonymization(validation_g1_file, validation_g2_file, 
                             seed_sample_file, output_file="pptmethod_mapping.txt",
                             threshold=0.5, max_iterations=20):
    """Uses the method provided within PPTs to map nodes"""
    try:
        print("\nMethod may take 10-15 minutes, please be patient.")

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

def neighbor_based_mapping(validation_g1_file, validation_g2_file, 
                         seed_sample_file,
                         output_file="neighbor_mapping.txt"):
    """
    Generates node mapping using:
    - 500 correct seed pairs
    - Neighbor similarity for remaining nodes
    - Falls back to degree matching when needed
    """
    try:
        # Read graphs
        G1 = nx.read_edgelist(validation_g1_file, nodetype=int)
        G2 = nx.read_edgelist(validation_g2_file, nodetype=int)
        
        # Get degree dictionaries
        g1_degrees = dict(G1.degree())
        g2_degrees = dict(G2.degree())
        
        # Read seed pairs
        with open(seed_sample_file, 'r') as f:
            seed_pairs = [list(map(int, line.strip().split())) for line in f if line.strip()]
            seed_dict = {g1: g2 for g1, g2 in seed_pairs}
        
        # Create reverse mapping (G2 -> G1)
        reverse_seed = {v: k for k, v in seed_dict.items()}
        
        # Find unmapped nodes
        mapped_g1 = set(seed_dict.keys())
        mapped_g2 = set(seed_dict.values())
        unmapped_g1 = list(set(G1.nodes()) - mapped_g1)
        unmapped_g2 = list(set(G2.nodes()) - mapped_g2)
        
        # Create degree bins for fallback
        degree_to_g2 = {}
        for node in unmapped_g2:
            deg = g2_degrees[node]
            if deg not in degree_to_g2:
                degree_to_g2[deg] = []
            degree_to_g2[deg].append(node)
        
        # Neighbor-based matching
        full_mapping = seed_dict.copy()
        for g1_node in unmapped_g1:
            best_match = None
            max_score = -1
            
            # Get G1's neighbors and their mappings
            g1_neighbors = set(G1.neighbors(g1_node))
            mapped_neighbors = [seed_dict[n] for n in g1_neighbors if n in seed_dict]
            
            # Only proceed if we have some mapped neighbors
            if mapped_neighbors:
                # Check all possible G2 candidates
                for g2_candidate in unmapped_g2:
                    g2_neighbors = set(G2.neighbors(g2_candidate))
                    
                    # Calculate overlap score
                    score = len(set(mapped_neighbors) & g2_neighbors)
                    
                    if score > max_score:
                        max_score = score
                        best_match = g2_candidate
            
            # Fallback to degree matching if no good neighbor match
            if best_match is None:
                target_deg = g1_degrees[g1_node]
                closest_deg = min(degree_to_g2.keys(), 
                                 key=lambda x: abs(x - target_deg))
                if degree_to_g2[closest_deg]:
                    best_match = degree_to_g2[closest_deg].pop()
            
            if best_match:
                full_mapping[g1_node] = best_match
                if best_match in unmapped_g2:
                    unmapped_g2.remove(best_match)
        
        # Write output
        with open(output_file, 'w') as f:
            for g1, g2 in full_mapping.items():
                f.write(f"{g1} {g2}\n")
        
        print(f"Generated {output_file} with {len(seed_dict)} seed pairs + {len(unmapped_g1)} neighbor-based mappings.")
    
    except Exception as e:
        print(f"Error in neighbor-based mapping: {e}")
        return None


def hybrid_deanonymization(validation_g1_file, validation_g2_file, 
                         seed_sample_file, output_file="hybrid_mapping.txt",
                         threshold=0.5, max_iterations=20):
    try:
        # 1. First run seed-based propagation
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
        print(f"Hybrid mapping complete! Final count: {final_count}/{len(G1.nodes())}")
        return mapped_pairs
        
    except Exception as e:
        print(f"Error in hybrid deanonymization: {e}")
        return None
    # Future Idea:
        # Just for fun but after the PPT Method we check and make sure all mapped seeds are correct and then if not it reruns, we can do this with lower thresholds as well.
        # This won't work on the actual code since we cannot check but it might be a good idea for testing purposes to see the lowest threshold which remains accurate.

def jaccard_similarity(u, v, G1, G2, mapped_pairs):
    """Calculate Jaccard similarity between neighborhoods"""
    # Get mapped neighbors of u in G1
    u_neighbors = {mapped_pairs[n] for n in G1.neighbors(u) if n in mapped_pairs}
    
    # Get neighbors of v in G2
    v_neighbors = set(G2.neighbors(v))
    
    intersection = len(u_neighbors & v_neighbors)
    union = len(u_neighbors | v_neighbors)
    
    return intersection / union if union > 0 else 0

def jaccard_hybrid_deanonymization(validation_g1_file, validation_g2_file, 
                                  seed_sample_file, output_file="jaccard_hybrid.txt",
                                  threshold=0.6, max_iterations=20):
    try:
        # [Basically the same as hybrid_deanonymization but replace score calculation being common neighbors w/ Jaccard Similarity]
        # 1. First run seed-based propagation
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

        # 3 Enhanced neighbor matching with Jaccard
        for u in unmapped_g1:
            best_match = None
            max_score = -1
            
            # Only proceed if node has mapped neighbors
            if any(n in mapped_pairs for n in G1.neighbors(u)):
                for v in unmapped_g2:
                    score = jaccard_similarity(u, v, G1, G2, mapped_pairs)
                    
                    # Additional degree similarity factor (0.5 weight)
                    degree_sim = 1 / (1 + abs(G1.degree(u) - G2.degree(v)))
                    combined_score = 0.7 * score + 0.3 * degree_sim
                    
                    if combined_score > max_score:
                        max_score = combined_score
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
        print(f"Hybrid mapping complete! Final count: {final_count}/{len(G1.nodes())}")
        return mapped_pairs
        
    
    except Exception as e:
        print(f"Error in enhanced hybrid deanonymization: {e}")
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

            # ppt_method_deanonymization(validation_g1, validation_g2, "validation_seed_sample.txt")
            # correct, total, acc = calculate_mapping_accuracy("pptmethod_mapping.txt", validation_seed_mapping)
            # print(f"PPT Method Accuracy: {correct}/{total} ({acc:.2f}%)")

            jaccard_hybrid_deanonymization(validation_g1, validation_g2, "validation_seed_sample.txt")
            correct, total, acc = calculate_mapping_accuracy("jaccard_hybrid.txt", validation_seed_mapping)
            print(f"Jaccard Hybrid Method Accuracy: {correct}/{total} ({acc:.2f}%)")

            hybrid_deanonymization(validation_g1, validation_g2, "validation_seed_sample.txt")
            correct, total, acc = calculate_mapping_accuracy("hybrid_mapping.txt", validation_seed_mapping)
            print(f"Hybrid Method Accuracy: {correct}/{total} ({acc:.2f}%)")
            

        else:
            print("\nPlease add the missing files and try again.")
    else:
        print("Currently not implemented.")

    

if __name__ == "__main__":
    main()