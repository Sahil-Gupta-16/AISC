# --- Fuzzy Braking System using Mamdani Inference ---
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Triangular Membership Function
# ---------------------------------------------------------
def trimf(x, a, b, c):
    if a == b:          # Left shoulder
        return 1.0 if x <= b else max(min((c - x) / (c - b), 1), 0)

    elif b == c:        # Right shoulder
        return 1.0 if x >= b else max(min((x - a) / (b - a), 1), 0)

    elif a < b < c:     # Normal triangle
        return max(min((x - a) / (b - a), (c - x) / (c - b)), 0)

    return 0  # fallback


# ---------------------------------------------------------
# Plotting Utility
# ---------------------------------------------------------
def plot_membership(universe, mf_dict, var_name):
    plt.figure()
    for label, points in mf_dict.items():
        plt.plot(universe, points, label=label)
    plt.title(f'Membership Functions for {var_name}')
    plt.xlabel(var_name)
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------
# Step 1: Define Universes
# ---------------------------------------------------------
speed_range = np.arange(0, 101, 1)
distance_range = np.arange(0, 101, 1)
brake_power_range = np.arange(0, 101, 1)


# ---------------------------------------------------------
# Step 2: Define Fuzzy Memberships
# ---------------------------------------------------------
speed_mfs = {
    'small': [trimf(x, 0, 0, 30) for x in speed_range],
    'medium': [trimf(x, 20, 40, 60) for x in speed_range],
    'high': [trimf(x, 50, 70, 90) for x in speed_range],
    'very_high': [trimf(x, 80, 100, 100) for x in speed_range]
}

distance_mfs = {
    'small': [trimf(x, 0, 0, 25) for x in distance_range],
    'medium': [trimf(x, 15, 35, 55) for x in distance_range],
    'high': [trimf(x, 45, 65, 85) for x in distance_range],
    'very_high': [trimf(x, 75, 100, 100) for x in distance_range]
}

brake_power_mfs = {
    'small': [trimf(x, 0, 0, 25) for x in brake_power_range],
    'medium': [trimf(x, 15, 40, 65) for x in brake_power_range],
    'high': [trimf(x, 55, 75, 90) for x in brake_power_range],
    'very_high': [trimf(x, 80, 100, 100) for x in brake_power_range]
}


# ---------------------------------------------------------
# Plot membership functions
# ---------------------------------------------------------
plot_membership(speed_range, speed_mfs, "Speed")
plot_membership(distance_range, distance_mfs, "Distance")
plot_membership(brake_power_range, brake_power_mfs, "Brake Power")


# ---------------------------------------------------------
# Step 4: Rule Base (Fuzzy Associative Matrix)
# ---------------------------------------------------------
FAM = {
    ('small', 'small'): 'small',
    ('small', 'medium'): 'small',
    ('small', 'high'): 'medium',
    ('small', 'very_high'): 'high',

    ('medium', 'small'): 'small',
    ('medium', 'medium'): 'medium',
    ('medium', 'high'): 'high',
    ('medium', 'very_high'): 'very_high',

    ('high', 'small'): 'medium',
    ('high', 'medium'): 'high',
    ('high', 'high'): 'very_high',
    ('high', 'very_high'): 'very_high',

    ('very_high', 'small'): 'high',
    ('very_high', 'medium'): 'very_high',
    ('very_high', 'high'): 'very_high',
    ('very_high', 'very_high'): 'very_high'
}


# ---------------------------------------------------------
# Step 5: Fuzzification
# ---------------------------------------------------------
def speed_fuzzification(speed):
    params = {
        'small': (0, 0, 30),
        'medium': (20, 40, 60),
        'high': (50, 70, 90),
        'very_high': (80, 100, 100)
    }
    degrees = {label: trimf(speed, *vals) for label, vals in params.items()}
    print(f"Fuzzification of speed={speed}: {degrees}")
    return degrees


def distance_fuzzification(distance):
    params = {
        'small': (0, 0, 25),
        'medium': (15, 35, 55),
        'high': (45, 65, 85),
        'very_high': (75, 100, 100)
    }
    degrees = {label: trimf(distance, *vals) for label, vals in params.items()}
    print(f"Fuzzification of distance={distance}: {degrees}")
    return degrees


# ---------------------------------------------------------
# Step 6: Rule Evaluation (Mamdani max–min)
# ---------------------------------------------------------
def apply_rules(speed_degrees, distance_degrees):
    output_memberships = {}
    print("\nRule evaluations and firing strengths:")

    for s_label, s_degree in speed_degrees.items():
        if s_degree > 0:
            for d_label, d_degree in distance_degrees.items():
                if d_degree > 0:
                    firing_strength = min(s_degree, d_degree)
                    output_label = FAM[(s_label, d_label)]
                    prev_strength = output_memberships.get(output_label, 0)
                    output_memberships[output_label] = max(firing_strength, prev_strength)

                    print(f"IF speed IS {s_label} ({s_degree:.2f}) AND distance IS {d_label} ({d_degree:.2f}) "
                          f"THEN brake_power IS {output_label} (strength={firing_strength:.2f})")

    print(f"\nCombined output memberships: {output_memberships}")
    return output_memberships


# ---------------------------------------------------------
# Step 7: Defuzzification (Centroid approximation)
# ---------------------------------------------------------
def brake_power_val(label):
    values = {'small': 20, 'medium': 50, 'high': 75, 'very_high': 100}
    return values[label]


def defuzzify(output_memberships):
    numerator = sum(brake_power_val(label) * degree for label, degree in output_memberships.items())
    denominator = sum(output_memberships.values())
    crisp_output = numerator / denominator if denominator != 0 else 0
    print(f"\nDefuzzified output (crisp brake power): {crisp_output:.2f}")
    return crisp_output


# ---------------------------------------------------------
# Full fuzzy controller
# ---------------------------------------------------------
def fuzzy_controller(speed_input, distance_input):
    sp_degrees = speed_fuzzification(speed_input)
    dist_degrees = distance_fuzzification(distance_input)
    output_mfs = apply_rules(sp_degrees, dist_degrees)
    return defuzzify(output_mfs)


# ---------------------------------------------------------
# Run Example
# ---------------------------------------------------------
speed_val = int(input("Enter speed (0-100): "))
distance_val = int(input("Enter distance (0-100): "))

print(f"\nRunning fuzzy controller for speed={speed_val}, distance={distance_val}\n")

brake_power_output = fuzzy_controller(speed_val, distance_val)

print(f"\n🚗 Final Brake Power Output: {brake_power_output:.2f}")
