# Changelog

## Unreleased

- Added `thunder_v3_cad_inertia.urdf`, a Thunder v3 URDF variant that keeps the
  existing training-compatible link and joint names while replacing all 21
  `<inertial>` blocks with the CAD-exported origin, mass, and inertia tensor
  values from the root-drop hardware asset.
- Updated Thunder asset configs to load `thunder_v3_cad_inertia.urdf` instead
  of the older Thunder URDF paths, including the previously missing
  `Robots/thunder/urdf/thunder.urdf` path.
