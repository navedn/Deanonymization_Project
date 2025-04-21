import os
import random
import networkx as nx

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

            generate_random_mapping(validation_g1, validation_g2, "validation_seed_sample.txt")
            correct, total, acc = calculate_mapping_accuracy("guessed_seed_mapping.txt", validation_seed_mapping)
            print(f"Guessed Accuracy: {correct}/{total} ({acc:.2f}%)")
        else:
            print("\nPlease fix the missing files and try again.")
    else:
        print("Currently not implemented.")

    

if __name__ == "__main__":
    main()