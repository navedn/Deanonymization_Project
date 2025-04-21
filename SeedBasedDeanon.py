import os

def main():
    print("Welcome to the Seed-Based De-anonymization Program.")
    print("If you have not done so already, place the following files are in the same directory as this script:")
    print("- G1 dataset file (e.g., seed_G1.edgelist)")
    print("- G2 dataset file (e.g., seed_G2.edgelist)")
    print("- Seed mapping file (e.g., seed_mapping.txt)\n")

    # Ask for filenames
    g1_file = input("Enter the name of the G1 dataset file (e.g., seed_G1.edgelist): ")
    g2_file = input("Enter the name of the G2 dataset file (e.g., seed_G2.edgelist): ")
    seed_file = input("Enter the name of the seed mapping file (e.g., seed_mapping.txt): ")

    # Check if files exist
    files = [g1_file, g2_file, seed_file]
    all_exist = True

    for file in files:
        if not os.path.exists(file):
            print(f"Error: File '{file}' not found in the current directory.")
            all_exist = False

    if all_exist:
        print("\nSuccess! All files are ready for processing.")
    else:
        print("\nPlease fix the missing files and try again.")

if __name__ == "__main__":
    main()