import os
import pandas as pd
from argparse import ArgumentParser



def walk_files_recursively(directory):
    """
    Print all files in the given directory and its subdirectories.

    Args:
        directory (str): The path to the directory to start from.
    """
    # Walk through all directories and files
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct and print the full path to each file
            file_path = os.path.join(root, file)
            paths.append(file_path)

    return paths


def parse_filepath(filepath):
    """
    Parse the given filepath to extract specific components.

    Args:
        filepath (str): The filepath to parse

    Returns:
        tuple: A tuple containing (roll_type, network_type, file_name)
    """
    # Split the path into components
    parts = filepath.split("/")

    # # Extract the roll type (assuming it's always in this position)
    # roll_type = parts[-3]

    # Extract the network type
    network_type = parts[-2]

    # Extract the file name without extension
    file_name = parts[-1].split(".")[0]

    # Extract run
    run = file_name[-1]

    return network_type, file_name, run


def parse_file_data(path):
    data = []
    with open(path, "r") as file:
        flag = False
        for line in file:
            if "runtime" in line:
                data.append(line)
            elif "compile time:" in line:
                data.append(line)
            elif "data size:" in line:
                data.append(line)
            elif "depth:" in line:
                data.append(line)
            elif "FHEOp.PACK" in line:
                data.append(line)
            elif "picked kernel" in line:
                flag = True
            elif flag:
                data.append(line)
                flag = False
    return data


def parse_data_to_dictionary(data_lines):
    """
    Parse the given data lines into a single dictionary, with entries from
    both dictionaries merged into the parent dictionary.

    Args:
        data_lines (list): List of strings containing the data to parse

    Returns:
        dict: A dictionary containing all the parsed data
    """
    result = {}

    for line in data_lines:
        line = line.strip()

        # Parse the first dictionary directly into result
        if line.startswith("{'add':"):
            continue
            # Remove curly braces and split by comma
            clean_line = line.strip("{}'\n")
            parts = clean_line.split(", ")

            for part in parts:
                key, value = part.split(": ")
                result[key] = int(value)

        # Parse FHEOp dictionary directly into result
        elif line.startswith("{'FHEOp"):
            # Extract the key-value pairs
            clean_line = line.strip("{}\n")
            parts = clean_line.split(", ")
            for part in parts:
                enum_part, value = part.split(": ")
                # Extract just the operation name from the enum
                op_name = enum_part.split("'")[1].split("'")[0].split(".")[-1]  # Get the string between quotes
                result[op_name] = int(value)
    
        # Parse simple key-value lines (compile time, comm cost, etc.)
        elif ": " in line:
            key, value = line.split(": ")
            try:
                # Try to convert to float if possible
                result[key] = float(value)
            except ValueError:
                # Keep as string if not a number
                result[key] = value
    return result


# Example usage
if __name__ == "__main__":
    # Replace this with your directory path

    parser = ArgumentParser()
    parser.add_argument("--sys")
    parser.add_argument("--fn")
    args = parser.parse_args()

    
    directory_path = f"./logs/{args.sys}"  # Current directory
    paths = walk_files_recursively(directory_path)

    results = []
    for path in paths:
        path_data = parse_filepath(path)
        data = parse_file_data(path)
        result = parse_data_to_dictionary(data)
        # result["roll_type"] = path_data[1]
        result["fn"] = path_data[1]
        result["run"] = path_data[2]

        if "SUB" in result:
            result["ADD"] += result["SUB"]
            del result["SUB"]        

        results.append(result)

    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(results)
    print(df)
    print()
    if args.sys == "rotom":
        filtered_df = df[df["fn"].str.startswith(args.fn)].sort_values("run")[["fn", "run", "runtime", "compile time", "data size", "depth"]]
    else:
        filtered_df = df[df["fn"].str.startswith(args.fn)].sort_values("run")[["fn", "run", "runtime", "data size", "depth"]]
    
    print(filtered_df)
    data = float(list(filtered_df["data size"])[0].split(" ")[0]) * 8
    lan_time = data * 1e-4
    wan_time = data * 1e-3
    if args.sys == "rotom":
        print("compile time:", filtered_df["compile time"].mean())
    print("runtime:", filtered_df["runtime"].mean())
    print("lan runtime:", filtered_df["runtime"].mean() + lan_time)
    print("wan runtime:", filtered_df["runtime"].mean() + wan_time)

