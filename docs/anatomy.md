# Anatomical structures

AnyCXR's packaged inference taxonomy contains **54 overlapping anatomical channels**. This catalog is generated from the [inference reference](../cxas/data/anychest_reference.json); the [JSON catalog](anatomy.json) includes the exact zero-based channel index for every output.

Left and right refer to the patient's anatomy. Grouping below is for browsing; the inference channel order is unchanged. These are output capabilities, not per-structure accuracy claims.

| Group | Channels |
| --- | ---: |
| Lungs | 5 |
| Ribs | 24 |
| Thoracic vertebrae | 11 |
| Heart and vessels | 9 |
| Other bones | 5 |

## Lungs

| Channel (zero-based) | Structure | Output name |
| ---: | --- | --- |
| 44 | Left upper lung lobe | `lung_upper_lobe_left` |
| 45 | Right upper lung lobe | `lung_upper_lobe_right` |
| 46 | Right middle lung lobe | `lung_middle_lobe_right` |
| 47 | Left lower lung lobe | `lung_lower_lobe_left` |
| 48 | Right lower lung lobe | `lung_lower_lobe_right` |

## Ribs

| Channel (zero-based) | Structure | Output name |
| ---: | --- | --- |
| 7 | Left rib 1 | `rib_left_1` |
| 8 | Left rib 2 | `rib_left_2` |
| 9 | Left rib 3 | `rib_left_3` |
| 10 | Left rib 4 | `rib_left_4` |
| 11 | Left rib 5 | `rib_left_5` |
| 12 | Left rib 6 | `rib_left_6` |
| 13 | Left rib 7 | `rib_left_7` |
| 14 | Left rib 8 | `rib_left_8` |
| 15 | Left rib 9 | `rib_left_9` |
| 16 | Left rib 10 | `rib_left_10` |
| 17 | Left rib 11 | `rib_left_11` |
| 18 | Left rib 12 | `rib_left_12` |
| 19 | Right rib 1 | `rib_right_1` |
| 20 | Right rib 2 | `rib_right_2` |
| 21 | Right rib 3 | `rib_right_3` |
| 22 | Right rib 4 | `rib_right_4` |
| 23 | Right rib 5 | `rib_right_5` |
| 24 | Right rib 6 | `rib_right_6` |
| 25 | Right rib 7 | `rib_right_7` |
| 26 | Right rib 8 | `rib_right_8` |
| 27 | Right rib 9 | `rib_right_9` |
| 28 | Right rib 10 | `rib_right_10` |
| 29 | Right rib 11 | `rib_right_11` |
| 30 | Right rib 12 | `rib_right_12` |

## Thoracic vertebrae

| Channel (zero-based) | Structure | Output name |
| ---: | --- | --- |
| 32 | Thoracic vertebra T2 | `vertebrae_T2` |
| 33 | Thoracic vertebra T3 | `vertebrae_T3` |
| 34 | Thoracic vertebra T4 | `vertebrae_T4` |
| 35 | Thoracic vertebra T5 | `vertebrae_T5` |
| 36 | Thoracic vertebra T6 | `vertebrae_T6` |
| 37 | Thoracic vertebra T7 | `vertebrae_T7` |
| 38 | Thoracic vertebra T8 | `vertebrae_T8` |
| 39 | Thoracic vertebra T9 | `vertebrae_T9` |
| 40 | Thoracic vertebra T10 | `vertebrae_T10` |
| 41 | Thoracic vertebra T11 | `vertebrae_T11` |
| 42 | Thoracic vertebra T12 | `vertebrae_T12` |

## Heart and vessels

| Channel (zero-based) | Structure | Output name |
| ---: | --- | --- |
| 0 | Left ventricle | `heart_ventricle_left` |
| 1 | Left atrium | `heart_atrium_left` |
| 2 | Right ventricle | `heart_ventricle_right` |
| 3 | Right atrium | `heart_atrium_right` |
| 4 | Myocardium | `heart_myocardium` |
| 5 | Pulmonary artery | `pulmonary_artery` |
| 6 | Whole heart | `heart` |
| 31 | Aorta (aorta channel) | `aorta` |
| 43 | Aorta (aorta_long channel) | `aorta_long` |

## Other bones

| Channel (zero-based) | Structure | Output name |
| ---: | --- | --- |
| 49 | Left clavicle | `clavicula_left` |
| 50 | Right clavicle | `clavicula_right` |
| 51 | Left humerus | `humerus_left` |
| 52 | Right humerus | `humerus_right` |
| 53 | Sternum | `sternum` |

`aorta` and `aorta_long` are separate legacy output names. The packaged reference does not specify a finer distinction, so neither is renamed as a specific aortic segment.

Projected anatomies overlap: keep independent channels rather than collapsing masks with argmax. Class names are the per-class PNG filenames written by the legacy CLI. A structure may be cropped, occluded, ambiguous or poorly predicted in a particular image.

Regenerate this page and the JSON with `python scripts/build_readme_assets.py`.
