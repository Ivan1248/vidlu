# Attribute baseline values for iRAP VLM classification

Reference: `vidlu_irap_gaim/vlm/attribute_prompts.yaml`

Definitions:
- **Most common**: The mode (highest-frequency value) expected on a typical road network; often the majority class in training data.
- **Most pessimistic**: The value that, if always predicted, would least underestimate risk for iRAP star rating (assumes worst safety conditions).

---

## Carriageway

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Carriageway label | Undivided road | — |

---

## Observed flows (exposure-related)

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Motorcycle observed flow | None | 8+ motorcycles |
| Bicycle observed flow | None | 8+ bicycles |
| Pedestrian observed flow across the road | None | 8+ pedestrians |
| Pedestrian observed flow along the road driver-side | None | 8+ pedestrians |
| Pedestrian observed flow along the road passenger-side | None | 8+ pedestrians |

---

## Speed & traffic calming

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Speed limit | 50km/h | — |
| Motorcycle speed limit | 50km/h | — |
| Truck Speed limit | 50km/h | — |
| Differential speed limits | Not present | Present |
| Speed management / traffic calming | Not present | Not present |

---

## Lane geometry

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Number of lanes | One | Four or more |
| Lane width | Medium 2.75m to <3.25m | Narrow 0m to <2.75m |

---

## Alignment

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Curvature | Straight or gently curving | Very sharp |
| Quality of curve | Not applicable | Poor |

---

## Infrastructure quality

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Upgrade cost | Low | High |
| Skid resistance / grip | Sealed – adequate | Unsealed – poor |
| Road condition | Good | Poor |
| Delineation | Adequate | Poor |
| Sight distance | Adequate | Poor |

---

## Median (head-on risk)

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Median Type | Centre line | Centre line |

---

## Roadside

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Vehicle parking | None | Two side |
| Grade | 0% to <7.5% | ≥10% |
| Roadworks | No road works | Major road works |
| Street lighting | Not present | Not present |
| Service road | Not present | Not present |
| Centreline rumble strips code | Not present | Present |
| Shoulder rumble strips | Not present | Present |

---

## Roadside severity

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Roadside severity - driver-side distance | 1 to <5m | 0 to <1m |
| Roadside severity - driver-side object | Tree ≥10cm | Cliff |
| Roadside severity - passenger-side distance | 1 to <5m | 0 to <1m |
| Roadside severity - passenger-side object | Tree ≥10cm | Cliff |

---

## Shoulders

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Paved shoulder - driver-side | Narrow 0m to <1m | None |
| Paved shoulder - passenger-side | Narrow 0m to <1m | None |

---

## Intersections

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Intersection type | None | 4-leg |
| Intersection channelisation | Not present | Not present |
| Intersection quality | Not applicable | Poor |
| Intersecting road volume | Not applicable | — |

---

## Property access & land use

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Property access points | None | Commercial access ≥1 |
| Land use - driver-side | Undeveloped areas | Educational |
| Land use - passenger-side | Undeveloped areas | Educational |
| Area type | Urban | Urban |

---

## Pedestrian facilities

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Pedestrian crossing - inspected road | No facility | No facility |
| Pedestrian crossing quality | Not applicable | Poor |
| Pedestrian crossing - side road | No facility | No facility |
| Pedestrian fencing | Not present | Present |
| Sidewalk - driver-side | None | None |
| Sidewalk - passanger-side | None | None |

---

## Motorcycle & bicycle facilities

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| Motorcycle facilities | None | None |
| Bicycle facility | None | None |

---

## School zone

| Attribute | Most Common | Most Pessimistic |
|-----------|-------------|------------------|
| School zone warning | Not applicable (no school at the location) | No school zone warning (school present) |
| School zone crossing supervisor | Not applicable (no school at the location) | School zone crossing supervisor not present |

---

## Notes

1. **Empirical most-common values** can be computed from the BiH training set:
   ```bash
   IRAP_HOME=~/projects/irap_home python -m vidlu_irap_gaim.tools.baseline_random --split train --mode most_common
   ```

2. **Sparse response schemes** (`SparseStandardResponseScheme`, `SparseIndexedResponseScheme`) treat class index 0 as default and omit it from responses. The first-listed value in each attribute's YAML `values` block corresponds to index 0.
