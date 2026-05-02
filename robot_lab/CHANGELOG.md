# Changelog

## Unreleased

- Added `thunder_v3_cad_inertia.urdf`, a Thunder v3 URDF variant that keeps the
  existing training-compatible link and joint names while using the corrected
  inertial parameters from the standalone `thunder_v3_assets` asset version
  `cad-2026-05-02-symmetric-wheel`.
- Aligned `thunder_v3_cad_inertia.urdf` with the standalone asset repository's
  primary `urdf/thunder_v3.urdf`, including the `48.79163 kg` total mass and
  symmetric `1.40377 kg` wheel/foot link inertials.
- Updated Thunder asset configs to load `thunder_v3_cad_inertia.urdf` instead
  of the older Thunder URDF paths, including the previously missing
  `Robots/thunder/urdf/thunder.urdf` path.
