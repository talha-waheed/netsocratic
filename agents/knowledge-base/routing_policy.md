# Routing Policy Implementation Guide

## Reachability

OSPF with `redistribute connected subnets` automatically makes every router's stub subnet
(100.0.X.0/24) reachable from all other routers. No extra configuration is needed for basic
reachability — it is always provided by the base OSPF setup.

---

## Waypointing

A **waypoint** is an intermediate router that traffic must pass through (mandatory) or
should prefer to pass through (preferred) on its way to the destination.

### Mandatory Waypointing

Traffic must **always** traverse router W on the way from source S to destination D.

**Technique**: On all routers along non-W paths, set the cost of links that bypass W to
a prohibitively high value (e.g., 1000). Keep the cost of links toward W at 1.

**Example** — Athens must reach 100.0.29.0/24 (London's stub) via London (mandatory):
- Athens's path options: Athens→Istanbul→...→London or Athens→Sofia→...→London
- Since London owns the destination subnet, traffic always terminates at London.
  The waypoint here means traffic must use London as the final hop, not bypass it.
- To enforce a specific intermediate waypoint (e.g., Paris before London):
  set high cost on all Paris→London bypass paths and low cost on paths through Paris.

### Preferred Waypointing

Traffic **should prefer** to go through W but may use other paths if W is unavailable or
the cost is higher.

**Technique**: Set costs so the W-inclusive path has a lower total metric than alternatives,
but not so high that alternatives are completely blocked.

---

## Load Balancing (ECMP)

OSPF performs **Equal-Cost Multi-Path (ECMP)** routing when multiple paths to the same
destination have identical total cost. The `maximum-paths 32` directive allows up to 32
simultaneous ECMP paths.

### Achieving Exactly N ECMP Paths

To load-balance traffic across exactly N paths:
1. Identify N distinct routes from source to destination.
2. Set OSPF link costs so those N routes all have the **same total metric**.
3. Set all other routes to a higher metric so they are not selected.

**Example** — Athens to 100.0.29.0/24 across 3 equal-cost paths:
- Path 1: Athens → Istanbul → Bucharest → ... → London   (total cost T)
- Path 2: Athens → Sofia → Budapest → ... → London       (total cost T)
- Path 3: Athens → Istanbul → ... → Paris → London       (total cost T)
- Set interface costs on Athens and intermediate routers to make all three paths equal.

### Counting Paths

A path's total metric is the **sum of all OSPF link costs** from source to destination
(not counting the stub network's cost since it's redistributed as external). Adjust costs
on intermediate links until the desired number of paths share the minimum metric.

---

## Redundancy

Redundancy is achieved through ECMP: if one link fails, OSPF reconverges and uses the
remaining equal-cost paths. Requesting "N load-balanced paths" is equivalent to requesting
N-way ECMP, giving N-fold redundancy.

---

## Interaction of Waypointing and Load Balancing

When an intent specifies **both** a waypoint and N ECMP paths:
1. First enforce the waypoint by blocking non-waypoint paths (high cost).
2. Among the waypoint-inclusive paths, find N with equal cost.
3. The N ECMP paths must all pass through the required waypoint.

If fewer than N waypoint-inclusive paths exist in the topology, the maximum achievable
ECMP is limited by the topology itself — document this in the `decision_summary.txt`.
