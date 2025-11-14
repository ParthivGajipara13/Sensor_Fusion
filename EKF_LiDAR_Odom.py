import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
V_TRUE = 1.0
DIST_TOTAL = 10.0
T_END = DIST_TOTAL / V_TRUE

# Odom (5 Hz) with slip
DT_ODOM = 0.2
SLIP_START_M, SLIP_END_M = 5.0, 6.0
SLIP_FACTOR = 0.7
VEL_NOISE_STD = 0.02

# LiDAR (2 Hz) absolute pose
RATE_LIDAR_HZ = 2.0
DT_LIDAR = 1.0 / RATE_LIDAR_HZ
POSE_XY_STD = 0.03
POSE_TH_STD = np.deg2rad(1.5)

# LiDAR (raw scan for Figure 2)
FOV_DEG, ANGLE_STEP_DEG = 270.0, 1.0
MAX_RANGE_M, MIN_RANGE_M = 25.0, 0.05
WALL_X = 12.0
RANGE_NOISE_STD = 0.02

np.random.seed(42)

# ---------------- Helpers ----------------
def save_df_json(df: pd.DataFrame, path: str):
    # Pretty JSON array of records
    df.to_json(path, orient="records", indent=2)

def load_df_json(path: str) -> pd.DataFrame:
    return pd.read_json(path)

# ---------------- 1) Generate odometry (5 Hz) ----------------
def generate_odom_df():
    t_odom = np.arange(0.0, T_END + 1e-12, DT_ODOM)
    x_true = V_TRUE * t_odom

    v = np.full_like(t_odom, V_TRUE, float)
    slip = (x_true >= SLIP_START_M) & (x_true <= SLIP_END_M)
    v[slip] *= SLIP_FACTOR
    if len(v) > 1:
        v[1:] += np.random.normal(0.0, VEL_NOISE_STD, len(v)-1)

    x_odom = np.zeros_like(v)
    for i in range(1, len(v)):
        x_odom[i] = x_odom[i-1] + v[i]*DT_ODOM

    odom = pd.DataFrame({
        "time_s": t_odom,
        "x_m": x_odom,
        "y_m": 0.0,
        "theta_rad": 0.0,
        "v_mps": v,
        "w_rps": 0.0,
    })
    return odom, t_odom, x_true

# ---------------- 2) Generate LiDAR poses (2 Hz) ----------------
def generate_lidar_pose_df():
    t_lidar = np.arange(0.0, T_END + 1e-12, DT_LIDAR)
    x_true_L = V_TRUE * t_lidar
    y_true_L = np.zeros_like(t_lidar)
    th_true_L = np.zeros_like(t_lidar)

    xL  = x_true_L + np.random.normal(0.0, POSE_XY_STD, t_lidar.size)
    yL  = y_true_L + np.random.normal(0.0, POSE_XY_STD, t_lidar.size)
    thL = th_true_L + np.random.normal(0.0, POSE_TH_STD, t_lidar.size)

    lidar_pose = pd.DataFrame({
        "time_s": t_lidar,
        "x_m": xL,
        "y_m": yL,
        "theta_rad": thL,
        "cov_xx": POSE_XY_STD**2, "cov_xy": 0.0, "cov_xt": 0.0,
        "cov_yx": 0.0,            "cov_yy": POSE_XY_STD**2, "cov_yt": 0.0,
        "cov_tx": 0.0,            "cov_ty": 0.0,            "cov_tt": POSE_TH_STD**2,
    })
    return lidar_pose

# ---------------- 3) EKF (predict with odom; update with LiDAR) ----------------
def run_ekf(odom: pd.DataFrame, lidar: pd.DataFrame) -> pd.DataFrame:
    odom = odom.sort_values("time_s").reset_index(drop=True)
    lidar = lidar.sort_values("time_s").reset_index(drop=True)

    x = np.array([0.0, 0.0, 0.0])                          # [x, y, theta]
    P = np.diag([0.5**2, 0.5**2, np.deg2rad(10.0)**2])     # initial cov

    sigma_v = 0.05   # m/s (process)
    sigma_w = 0.02   # rad/s

    rows = []
    i_o = i_l = 0
    t_prev = odom["time_s"].iloc[0]

    while i_o < len(odom) or i_l < len(lidar):
        next_ot = odom["time_s"].iloc[i_o] if i_o < len(odom) else np.inf
        next_lt = lidar["time_s"].iloc[i_l] if i_l < len(lidar) else np.inf

        if next_ot <= next_lt:
            # Predict
            t = next_ot
            dt = max(1e-9, t - t_prev)
            v = float(odom["v_mps"].iloc[i_o])
            w = float(odom["w_rps"].iloc[i_o])

            th = x[2]
            x[0] += v*dt*np.cos(th)
            x[1] += v*dt*np.sin(th)
            x[2] += w*dt
            x[2] = (x[2] + np.pi) % (2*np.pi) - np.pi

            F = np.array([[1,0,-v*dt*np.sin(th)],
                          [0,1, v*dt*np.cos(th)],
                          [0,0, 1]])
            G = np.array([[dt*np.cos(th), 0],
                          [dt*np.sin(th), 0],
                          [0, dt]])
            Q = G @ np.diag([sigma_v**2, sigma_w**2]) @ G.T
            P = F @ P @ F.T + Q

            rows.append([t, x[0], x[1], x[2], "predict"])
            t_prev = t; i_o += 1
        else:
            # Update
            t = next_lt
            z = np.array([
                float(lidar["x_m"].iloc[i_l]),
                float(lidar["y_m"].iloc[i_l]),
                float(lidar["theta_rad"].iloc[i_l]),
            ])
            H = np.eye(3)
            R = np.array([
                [lidar["cov_xx"].iloc[i_l], lidar["cov_xy"].iloc[i_l], lidar["cov_xt"].iloc[i_l]],
                [lidar["cov_yx"].iloc[i_l], lidar["cov_yy"].iloc[i_l], lidar["cov_yt"].iloc[i_l]],
                [lidar["cov_tx"].iloc[i_l], lidar["cov_ty"].iloc[i_l], lidar["cov_tt"].iloc[i_l]],
            ], dtype=float)

            y = z - (H @ x)
            y[2] = (y[2] + np.pi) % (2*np.pi) - np.pi
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            x[2] = (x[2] + np.pi) % (2*np.pi) - np.pi
            P = (np.eye(3) - K @ H) @ P

            rows.append([t, x[0], x[1], x[2], "update"])
            t_prev = t; i_l += 1

    fused = pd.DataFrame(rows, columns=["time_s","x_m","y_m","theta_rad","step"])
    return fused

# ---------------- 4) LiDAR scan generator for Figure 2 ----------------
angles_deg = np.arange(-FOV_DEG/2, FOV_DEG/2 + 1e-12, ANGLE_STEP_DEG)
angles_rad = np.deg2rad(angles_deg)
t_scans = np.arange(0.0, T_END + 1e-12, DT_LIDAR)  # 2 Hz

def make_scan_points(t):
    x_world = V_TRUE * t
    ranges = []
    for a in angles_rad:
        if np.cos(a) > 1e-9:
            r_hit = (WALL_X - x_world)/np.cos(a)
            r = r_hit if r_hit > 0 else MAX_RANGE_M
        else:
            r = MAX_RANGE_M
        r = np.clip(r + np.random.normal(0.0, RANGE_NOISE_STD), MIN_RANGE_M, MAX_RANGE_M)
        ranges.append(r)
    r = np.array(ranges)
    xs, ys = r*np.cos(angles_rad), r*np.sin(angles_rad)  # LiDAR frame
    return xs, ys

# ---------------- Main ----------------
if __name__ == "__main__":
    # Generate & SAVE JSON
    odom_df, t_odom, x_true = generate_odom_df()
    lidar_df = generate_lidar_pose_df()
    save_df_json(odom_df, "odom.json")
    save_df_json(lidar_df, "lidar_pose.json")
    print("Saved odom.json and lidar_pose.json")

    # LOAD JSON and run EKF
    odom_loaded = load_df_json("odom.json")
    lidar_loaded = load_df_json("lidar_pose.json")
    fused_df = run_ekf(odom_loaded, lidar_loaded)
    save_df_json(fused_df, "fused.json")
    print("Saved fused.json")

    # ---------------- Figure 1: time vs X (incl. EKF) ----------------
    # (reconstruct fused X-time arrays)
    t_fused = fused_df["time_s"].values
    x_fused = fused_df["x_m"].values

    plt.figure(figsize=(12,5))
    plt.plot(t_odom, x_true, 'k--', label="True distance")
    plt.plot(t_odom, odom_df["x_m"], 'r-', label="Wheel odometry (with slip)")
    plt.scatter(t_odom, odom_df["x_m"], s=15, color='red', label="Odometry samples (5 Hz)")
    plt.scatter(lidar_df["time_s"], lidar_df["x_m"], s=40, color='blue',
                edgecolors='black', zorder=3, label="LiDAR scan positions (2 Hz)")
    plt.plot(t_fused, x_fused, linewidth=2.0, color='tab:green', label="EKF fused position")
    plt.axvspan(SLIP_START_M/V_TRUE, SLIP_END_M/V_TRUE, color='gray', alpha=0.15, label="Slip region")
    plt.title("Odom (5 Hz) & LiDAR (2 Hz) with EKF — X position vs time")
    plt.xlabel("Time [s]"); plt.ylabel("X position [m]")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # ---------------- Figure 2: LiDAR scans at start/mid/end ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14,5), sharex=False, sharey=False)
    scan_ids = [0, len(t_scans)//2, len(t_scans)-1]
    for ax, sid in zip(axes, scan_ids):
        xs, ys = make_scan_points(t_scans[sid])
        ax.scatter(xs, ys, s=6)
        ax.plot(0, 0, 'ko', markersize=5)  # rover/LiDAR at origin
        ax.set_aspect('equal'); ax.grid(True)
        ax.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
        ax.set_ylim(-MAX_RANGE_M, MAX_RANGE_M)
        ax.set_title(f"Scan {sid} (t={t_scans[sid]:.1f}s)")
        ax.set_xlabel("X [m]")
        if ax is axes[0]:
            ax.set_ylabel("Y [m]")

    fig.suptitle("LiDAR scans toward wall (2 Hz)", y=1.02, fontsize=14)
    plt.tight_layout(); plt.show()
