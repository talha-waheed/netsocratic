# Ground-Truth Rules

This document defines the general ground-truth format for network policy specifications.

## Purpose

Ground truth is the correct structured policy output for a given network scenario and requirement. It captures the intended semantics of the requirement, not raw router configuration text.

## Canonical Structure

The ground truth is a JSON object with three top-level sections:

```json
{
  "reachability": {
    "<source_router>": ["<prefix>", "..."]
  },
  "waypoint": {
    "(<source_router>,<prefix>)": ["<waypoint_router>", "..."]
  },
  "loadbalancing": {
    "(<source_router>,<prefix>)": <path_count>
  }
}
```

## Semantics

### Reachability

`reachability` declares which destination prefixes must be reachable from a given source router.

Example:

```json
{
  "reachability": {
    "istanbul": ["100.0.4.0/24"]
  },
  "waypoint": {},
  "loadbalancing": {}
}
```

Meaning: traffic from `istanbul` must be able to reach `100.0.4.0/24`.

### Waypoint

`waypoint` declares mandatory intermediate routers for traffic between a source router and a destination prefix.

Example:

```json
{
  "reachability": {
    "paris": ["100.0.4.0/24"]
  },
  "waypoint": {
    "(paris,100.0.4.0/24)": ["basel"]
  },
  "loadbalancing": {}
}
```

Meaning: traffic from `paris` to `100.0.4.0/24` must pass through `basel`.

### Load Balancing

`loadbalancing` declares the required number of paths for traffic between a source router and a destination prefix.

Example:

```json
{
  "reachability": {
    "athens": ["100.0.29.0/24"]
  },
  "waypoint": {
    "(athens,100.0.29.0/24)": ["london"]
  },
  "loadbalancing": {
    "(athens,100.0.29.0/24)": 3
  }
}
```

Meaning: traffic from `athens` to `100.0.29.0/24` must be distributed across `3` paths.

## General Rules

- `reachability` is the base connectivity specification.
- `waypoint` refines a source-prefix traffic demand with mandatory path constraints.
- `loadbalancing` refines a source-prefix traffic demand with a required path count.
- A source-prefix pair is written as `(<source_router>,<prefix>)`.
- Prefixes use CIDR notation such as `100.0.4.0/24`.
- `loadbalancing` values are integers.

## Practical Interpretation

To interpret a ground-truth object:

1. Read `reachability` as the required source-to-prefix connectivity set.
2. Read `waypoint` as mandatory traversal constraints on specific source-prefix pairs.
3. Read `loadbalancing` as path multiplicity constraints on specific source-prefix pairs.

## Compact Rule

Ground truth is the correct structured policy specification for a requirement. It encodes required reachability, optional waypoint constraints, and optional load-balancing constraints for source-to-prefix traffic demands.
