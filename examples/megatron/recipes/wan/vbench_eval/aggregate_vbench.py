import argparse
import json


# 1. Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Aggregate VBench results and calculate quality, semantic, and total scores."
)
parser.add_argument("json_file", type=str, help="Path to the JSON file containing VBench results")
parser.add_argument("--no-normalize", action="store_true", help="Skip normalization (use raw scores)")
args = parser.parse_args()

# 2. Load JSON data from file
with open(args.json_file, "r") as f:
    data = json.load(f)

# 3. Define Official Weights (based on VBench/VBench_Leaderboard/constants.py)
# Note: Dynamic Degree is the only metric with a 0.5 weight.
quality_weights = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "temporal_flickering": 1.0,
    "motion_smoothness": 1.0,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
    "dynamic_degree": 0.5,
}

semantic_weights = {
    "object_class": 1.0,
    "multiple_objects": 1.0,
    "human_action": 1.0,
    "color": 1.0,
    "spatial_relationship": 1.0,
    "scene": 1.0,
    "appearance_style": 1.0,
    "temporal_style": 1.0,
    "overall_consistency": 1.0,
}

# Normalization ranges (from VBench official implementation)
NORMALIZE_DIC = {
    "subject_consistency": {"Min": 0.1462, "Max": 1.0},
    "background_consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal_flickering": {"Min": 0.6293, "Max": 1.0},
    "motion_smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic_degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic_quality": {"Min": 0.0, "Max": 1.0},
    "imaging_quality": {"Min": 0.0, "Max": 1.0},
    "object_class": {"Min": 0.0, "Max": 1.0},
    "multiple_objects": {"Min": 0.0, "Max": 1.0},
    "human_action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial_relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance_style": {"Min": 0.0009, "Max": 0.2855},
    "temporal_style": {"Min": 0.0, "Max": 0.364},
    "overall_consistency": {"Min": 0.0, "Max": 0.364},
}


# 4. Normalize the data (if not disabled)
def normalize_scores(data, normalize_dic):
    """Normalize raw scores to [0, 1] range using official VBench bounds."""
    normalized_data = {}
    for metric, value in data.items():
        if metric in normalize_dic:
            min_val = normalize_dic[metric]["Min"]
            max_val = normalize_dic[metric]["Max"]
            normalized_value = (value - min_val) / (max_val - min_val)
            # Clamp to [0, 1] in case values are outside the expected range
            normalized_data[metric] = max(0.0, min(1.0, normalized_value))
        else:
            # If no normalization range defined, use raw value
            normalized_data[metric] = value
    return normalized_data


if not args.no_normalize:
    print("Applying normalization (use --no-normalize to skip)...")
    data = normalize_scores(data, NORMALIZE_DIC)
else:
    print("Using raw scores (no normalization)...")


# 5. Calculation Functions
def calculate_sub_score(data, weights):
    total_score = 0
    total_weight = 0
    for metric, weight in weights.items():
        if metric in data:
            total_score += data[metric] * weight
            total_weight += weight
        else:
            print(f"Warning: Metric '{metric}' not found in data.")
    return total_score / total_weight if total_weight > 0 else 0


# 6. Compute Scores
quality_score = calculate_sub_score(data, quality_weights)
semantic_score = calculate_sub_score(data, semantic_weights)

# The Official Total Score weights Quality 4x more than Semantic
total_score = (4 * quality_score + 1 * semantic_score) / 5

# 7. Output
print()
print(f"{'Metric':<25} | {'Score':<10}")
print("-" * 40)
print(f"{'Quality Score':<25} | {quality_score:.4f}")
print(f"{'Semantic Score':<25} | {semantic_score:.4f}")
print("-" * 40)
print(f"{'TOTAL SCORE':<25} | {total_score:.4f}")
