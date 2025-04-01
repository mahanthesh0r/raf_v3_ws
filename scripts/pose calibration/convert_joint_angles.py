import json
import math
import yaml

def read_and_convert_json(file_paths, config_path):
    joint_positions_dict = {}
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Navigate through the nested structure to access joint angles
        joint_angles_data = data["jointAnglesGroup"]["jointAngles"][0]["reachJointAngles"]["jointAngles"]["jointAngles"]
        
        # Extract values and convert degrees to radians
        if not file_path == "cup_feed_pose.json":
            joint_positions_radians = [entry["value"] for entry in joint_angles_data]
        else:
            joint_positions_radians = [math.radians(entry["value"]) for entry in joint_angles_data]

        # Simplify the key by removing the ".json" extension
        key = file_path.replace(".json", "")
        joint_positions_dict[key] = joint_positions_radians
        
        #print(f"Joint positions (radians) for {file_path}: {joint_positions_radians}")

    with open(config_path, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    config_data["joint_positions"] = joint_positions_dict


    with open(config_path, 'w') as yaml_file:
        yaml.safe_dump(config_data, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"Updated config file")


# Example usage
file_paths = [
    "bite_transfer.json",  # Change to your actual file path
    "multi_bite_transfer.json",  # Change to your actual file path
    "cup_feed_pose.json"  # Change to your actual file path
]
read_and_convert_json(file_paths, "/home/labuser/raf_v3_ws/src/raf_v3/scripts/config/config.yaml")