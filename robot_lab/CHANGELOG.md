# Changelog

## 2026-05-02 - cad-2026-05-02-symmetric-wheel-0131-limits

This release updates the active Thunder v3 simulation asset from the previous
primary URDF to a CAD-derived inertial model while preserving the RobotLab
training interface.

Active files:

- `urdf/thunder_v3.urdf`: primary RobotLab/Isaac Lab asset
- `xml/thunder_v3.xml`: synchronized XML-extension copy
- `urdf/legacy/thunder_v3_cad_2026-05-02.urdf`: raw CAD export kept for audit
- `urdf/legacy/thunder_0131copy.urdf`: older January reference kept for history

### Summary

- Total mass changed from `44.47163 kg` to `48.79163 kg`.
- Four wheel/foot links changed from `0.32377 kg` each to `1.40377 kg` each.
- Wheel/foot COM values were made left/right symmetric:
  - right wheels: `0.00000 -0.01385 0.00000`
  - left wheels: `0.00000 0.01385 0.00000`
- Wheel/foot inertia tensors were made symmetric:
  `ixx=0.00092`, `ixy=0.00000`, `ixz=0.00000`,
  `iyy=0.00121`, `iyz=0.00000`, `izz=0.00092`.
- `FR_calf`, `RR_calf`, and `RR_thigh` received CAD-derived COM and tensor
  corrections.
- Canonical RobotLab names, mesh paths, leg joint effort/velocity metadata,
  and continuous wheel joints were preserved.
- Leg joint position limits were aligned with the legacy 0131 reference:
  hips use `[-0.4, 0.4]`; `RR_calf_joint` and `RL_calf_joint` use
  `[-3.14, 3.14]`.
- The raw CAD URDF was archived, but it is not the active training asset.

### Detailed Inertial Changes

The table compares this version against the previous primary
`urdf/thunder_v3.urdf`.

| Link | Mass before -> after | COM before -> after | Inertia tensor before -> after (`ixx ixy ixz iyy iyz izz`) |
| --- | --- | --- | --- |
| `FR_calf` | `1.89384` -> `1.89384` | `0.17266 -0.02895 0.04224` -> `0.17379 -0.02895 0.03734` | `0.00844 0.01065 -0.01871 0.08429 0.00261 0.08034` -> `0.00744 0.01072 -0.01665 0.08429 0.00231 0.08134` |
| `FR_foot` | `0.32377` -> `1.40377` | `0.00146 -0.01336 -0.00634` -> `0.00000 -0.01385 0.00000` | `0.00072 0.00001 0.00000 0.00122 -0.00003 0.00071` -> `0.00092 0.00000 0.00000 0.00121 0.00000 0.00092` |
| `FL_foot` | `0.32377` -> `1.40377` | `0.00000 0.01336 0.00000` -> `0.00000 0.01385 0.00000` | `0.00071 0.00000 0.00000 0.00121 0.00000 0.00071` -> `0.00092 0.00000 0.00000 0.00121 0.00000 0.00092` |
| `RR_thigh` | `2.40982` -> `2.40982` | `0.19473 -0.02855 -0.01335` -> `0.19483 -0.02855 -0.01178` | `0.00596 0.01503 0.00785 0.12002 -0.00105 0.11954` -> `0.00584 0.01503 0.00694 0.12002 -0.00093 0.11966` |
| `RR_calf` | `1.89384` -> `1.89384` | `-0.17353 -0.02895 0.03877` -> `-0.17384 -0.02895 0.03735` | `0.00772 -0.01070 0.01726 0.08434 0.00240 0.08111` -> `0.00744 -0.01072 0.01666 0.08434 0.00231 0.08139` |
| `RR_foot` | `0.32377` -> `1.40377` | `0.00000 -0.01336 0.00000` -> `0.00000 -0.01385 0.00000` | `0.00071 0.00000 0.00000 0.00121 0.00000 0.00071` -> `0.00092 0.00000 0.00000 0.00121 0.00000 0.00092` |
| `RL_foot` | `0.32377` -> `1.40377` | `0.00000 0.01336 0.00000` -> `0.00000 0.01385 0.00000` | `0.00071 0.00000 0.00000 0.00121 0.00000 0.00071` -> `0.00092 0.00000 0.00000 0.00121 0.00000 0.00092` |

### Compatibility Notes

The raw CAD export was not used directly because it changes fields that
RobotLab depends on:

- lowercase link names with `_Link` suffixes
- package mesh paths instead of repository-relative mesh paths
- zero leg joint effort/velocity metadata
- wheel limit tags that are unsafe for continuous rolling-wheel joints

### Validation

Validated:

- parsed `urdf/thunder_v3.urdf`, `xml/thunder_v3.xml`, and the raw CAD archive
  as XML
- verified `21` links and `20` joints
- verified total mass `48.79163 kg`
- verified `urdf/thunder_v3.urdf` and `xml/thunder_v3.xml` are synchronized
- verified continuous wheel joints, wheel axes, and leg joint
  effort/velocity metadata
- verified legacy 0131 leg position limits while keeping current
  effort/velocity metadata
- verified the RobotLab copy
  `data/Robots/thunder_v3_assets/urdf/thunder_v3_cad_inertia.urdf` uses the
  same 21 inertial blocks while keeping RobotLab-specific asset packaging

Not validated:

- Isaac Lab runtime import after the GitHub push
- policy rollout with the updated dynamics
