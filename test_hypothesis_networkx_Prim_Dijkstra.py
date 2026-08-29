"""
# =============================================================================
# Property-Based Testing of NetworkX Graph Algorithms
# =============================================================================
# Authors      : Aswin S (SR No: 26240), Nitish Kumar Yadav (SR No: 26113)
# Algorithms   : Prim's Minimum Spanning Tree · Dijkstra's Shortest Path
# Framework    : Hypothesis (property-based testing) + pytest
# Library      : NetworkX 3.2+
# =============================================================================
#
# This module contains 10 property-based tests  which verifies fundamental and
# the correctness of Prim's MST and Dijkstra's shortest-path implementations
# in NetworkX.  
# All graph inputs are generated via Hypothesis composite
# strategies to enable automatic shrinking and full reproducibility.
#
# How to Run:  pytest -v test_hypothesis_networkx_Prim_Dijkstra.py
# =============================================================================
"""

import math
import sys

import networkx as nx
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


# =============================================================================
# Graph Generation Strategies
# =============================================================================


@st.composite
def connected_weighted_undirected_graph(
    draw, min_nodes: int = 2, max_nodes: int = 15
):
    """
    Hypothesis composite strategy that generates random connected, undirected,
    positively-weighted graphs.

    Construction
    ────────────
    1. Draw a node count *n* uniformly from [min_nodes, max_nodes].
    2. Build a guaranteed spanning path by chaining nodes in a randomly
       drawn permutation: perm[0]–perm[1]–…–perm[n−1].  This ensures
       connectivity regardless of *n*.
    3. For every remaining candidate edge, independently flip a fair coin
       and, if heads, draw a strictly positive weight and add the edge.

    Edge weights are drawn from the interval [0.01, 100.0] to keep all
    weights strictly positive, avoiding degenerate zero-weight ties that
    complicate comparison and ensuring Dijkstra's non-negative weight
    precondition is satisfied.

    The resulting graphs range from near-tree (sparse) to near-complete
    (dense), exercising a wide variety of structural configurations.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Step 2 — guarantee connectivity via a random-order spanning path
    perm = draw(st.permutations(range(n)))
    for i in range(n - 1):
        w = draw(
            st.floats(
                min_value=0.01,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        G.add_edge(perm[i], perm[i + 1], weight=w)

    # Step 3 — randomly add extra edges
    for u in range(n):
        for v in range(u + 1, n):
            if not G.has_edge(u, v) and draw(st.booleans()):
                w = draw(
                    st.floats(
                        min_value=0.01,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                G.add_edge(u, v, weight=w)

    return G


@st.composite
def connected_weighted_graph_with_node_pair(
    draw, min_nodes: int = 2, max_nodes: int = 15
):
    """
    Wraps ``connected_weighted_undirected_graph`` and additionally draws two
    nodes (possibly equal) from the generated graph, returning the triple
    ``(G, u, v)``.

    Tests that require distinct nodes apply ``assume(u != v)`` internally.
    Using ``st.sampled_from`` instead of Python's ``random`` module ensures
    Hypothesis can shrink and replay every generated example deterministically.
    """
    G = draw(
        connected_weighted_undirected_graph(
            min_nodes=min_nodes, max_nodes=max_nodes
        )
    )
    nodes = list(G.nodes())
    u = draw(st.sampled_from(nodes))
    v = draw(st.sampled_from(nodes))
    return G, u, v


# =============================================================================
#  PRIM'S  MINIMUM  SPANNING  TREE
#  Property-Based Tests
# =============================================================================


@given(G=connected_weighted_undirected_graph())
def test_prim_edge_count(G):
    """
    Property
    ────────
    The MST of a connected graph with *n* nodes has exactly *n − 1* edges.

    Property type: Postcondition / structural invariant.

    Why this property matters
    ─────────────────────────
    The edge-count invariant is the most fundamental structural requirement
    of a spanning tree.  By the tree characterisation theorem, a connected
    acyclic graph on *n* vertices has exactly *n − 1* edges.  Fewer edges
    would leave at least one vertex unreachable (the result is not
    spanning); more edges would introduce at least one cycle (the result
    is not a tree).

    Mathematical reasoning
    ──────────────────────
    Theorem: A connected graph T on n vertices is a tree if and only if
    |E(T)| = n − 1.  Proof sketch — induction on n: a single vertex has
    0 edges; adding one vertex to a tree to keep it connected adds
    exactly one edge.  Therefore any spanning tree returned by Prim's
    algorithm must satisfy |E(MST)| = |V(G)| − 1.

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes and strictly
    positive edge weights.  Density varies from near-tree (sparse) to
    near-complete (dense) because extra edges are added probabilistically
    by the Hypothesis strategy.

    Assumptions / preconditions
    ───────────────────────────
    • The input graph is connected (guaranteed by the generation strategy).
    • The graph has at least 2 nodes.

    How a failure indicates a bug
    ─────────────────────────────
    • Edge count < n − 1 → the algorithm dropped a vertex, so the result
      is not spanning.  This indicates a bug in node being visited or
      priority-queue processing.
    • Edge count > n − 1 → the algorithm kept a redundant edge that
      creates a cycle.  This indicates a bug in the cut-edge selection
      logic that should prevent revisiting already-connected components.

    Reference graph
    ───────────────
    See ``reference_graphs/test_prim_edge_count.png``
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert mst.number_of_edges() == G.number_of_nodes() - 1, (
        f"MST has {mst.number_of_edges()} edges but expected "
        f"{G.number_of_nodes() - 1} for a {G.number_of_nodes()}-node graph"
    )


@given(G=connected_weighted_undirected_graph())
def test_prim_connectivity_and_spanning(G):
    """
    Property
    ────────
    The MST is connected and its vertex set equals the vertex set of the
    original graph.

    Property type: Postcondition / definition-based invariant.

    Why this property matters
    ─────────────────────────
    A spanning tree must satisfy two conditions simultaneously:
    (a) it is *spanning* — it contains every vertex of G, and
    (b) it is *connected* — there is a path between every pair of
    vertices using only MST edges.
    Together with acyclicity, these conditions define a spanning tree.
    Failing either condition means the output is fundamentally invalid.

    Mathematical reasoning
    ──────────────────────
    Prim's algorithm initialises from an arbitrary root and grows a
    subtree by iterating the greedy cut-property step: at each iteration
    it selects the minimum-weight edge crossing the cut between the
    visited set S and V \\ S.  For a connected input, this process
    terminates only when S = V, so every vertex must appear in the MST
    and the result is connected by construction.  Any deviation from
    this indicates a bug in the traversal or termination logic.

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes.  Edge density
    ranges from sparse (near-tree) to dense (near-complete), so the
    test exercises the algorithm under varying amounts of edge competition
    and cut choices.

    Assumptions / preconditions
    ───────────────────────────
    • Input graph is connected (guaranteed by strategy).
    • All edge weights are strictly positive.

    How a failure indicates a bug
    ─────────────────────────────
    • MST is not connected → the algorithm returned a forest instead of
      a tree.  This indicates premature termination or a bug in the
      priority queue that fails to bridge all components.
    • Vertex sets differ → a vertex was silently dropped (or an extra
      phantom vertex was introduced).  This points to a bookkeeping bug
      in node tracking during tree construction.

    Reference graph
    ───────────────
    See ``reference_graphs/test_prim_connectivity_and_spanning.png``
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert nx.is_connected(mst), "MST is not connected — result is a forest"
    assert set(mst.nodes()) == set(G.nodes()), (
        f"MST nodes {set(mst.nodes())} ≠ original nodes {set(G.nodes())}"
    )


@given(G=connected_weighted_undirected_graph())
def test_prim_subgraph(G):
    """
    Property
    ────────
    Every edge in the MST exists in the original graph G — i.e., the MST
    is a subgraph of G.

    Property type: Postcondition / subgraph invariant.

    Why this property matters
    ─────────────────────────
    Prim's algorithm selects edges exclusively from G's edge set at each
    greedy step.  By definition, a spanning tree of G must be a
    *spanning subgraph*: same vertex set, edge set ⊆ E(G).  An MST edge
    that does not exist in G is structurally impossible and would make
    the result unrealisable in the original network.

    Mathematical reasoning
    ──────────────────────
    At every iteration of Prim's algorithm, the candidate edges are those
    incident to the current frontier of the growing tree — all of which
    belong to E(G).  The algorithm picks the minimum-weight candidate and
    adds it to the MST.  Since only edges from E(G) are ever considered,
    E(MST) ⊆ E(G) must hold.

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes and varying
    densities.  Dense graphs supply many candidate edges per iteration,
    testing whether the algorithm correctly restricts itself to the
    original edge set even when the frontier is large.

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected weighted graph.

    How a failure indicates a bug
    ─────────────────────────────
    An MST edge absent from G means the algorithm fabricated an edge —
    for example, by confusing node identifiers, corrupting adjacency
    data, or using a stale edge reference after graph mutation.  This
    is a critical correctness bug: the MST is no longer realisable
    within the original network.

    Reference graph
    ───────────────
    See ``reference_graphs/test_prim_subgraph.png``
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    for u, v in mst.edges():
        assert G.has_edge(u, v), (
            f"MST contains edge ({u}, {v}) which does not exist in G"
        )


@given(G=connected_weighted_undirected_graph())
def test_prim_acyclicity(G):
    """
    Property
    ────────
    The MST contains no cycles.

    Property type: Postcondition / structural invariant.

    Why this property matters
    ─────────────────────────
    Acyclicity is one of the two defining characteristics of a tree
    (together with connectivity).  A tree on n vertices has exactly n − 1
    edges; adding any edge creates exactly one cycle.  If the output
    contains a cycle, it is not a tree and cannot be a *minimum spanning
    tree*.

    Mathematical reasoning
    ──────────────────────
    Prim's algorithm enforces acyclicity via the cut property: at each
    step it adds the minimum-weight edge crossing the cut between the
    visited subtree S and V \\ S.  Since one endpoint is already in S
    and the other is not, the new edge cannot close a cycle within S.
    Therefore, if the implementation adheres to the cut property, the
    result is always a forest that is also connected (hence a tree).

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes.  Dense graphs
    (near-complete) are the hardest cases because G itself contains many
    cycles; the algorithm must avoid selecting any edge that would form
    one in the MST.

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected weighted graph.

    How a failure indicates a bug
    ─────────────────────────────
    A non-empty cycle basis means the algorithm added a redundant edge
    that connects two vertices already in the same connected component of
    the growing tree.  This indicates a bug in the visited-set check or
    priority-queue key update — the algorithm failed to recognise that
    both endpoints were already reachable, creating a cycle instead of
    a tree.

    Reference graph
    ───────────────
    See ``reference_graphs/test_prim_acyclicity.png``
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    cycles = nx.cycle_basis(mst)
    assert cycles == [], (
        f"MST contains {len(cycles)} cycle(s) — result is not a tree"
    )


@given(G=connected_weighted_undirected_graph())
def test_prim_weight_equals_kruskal(G):
    """
    Property
    ────────
    The total MST weight produced by Prim's algorithm equals the total
    weight produced by Kruskal's algorithm on the same graph.

    Property type: Metamorphic / cross-algorithm oracle.

    Why this property matters
    ─────────────────────────
    Directly verifying that an MST has minimum total weight would require
    enumerating all spanning trees — an operation that is exponential in
    complexity.  Instead, we use a *metamorphic* test: both Prim's and
    Kruskal's algorithms are provably optimal, so their total weights must
    agree.  Disagreement implies at least one algorithm returned a
    suboptimal result.

    Mathematical reasoning
    ──────────────────────
    The MST weight uniqueness theorem states that the total weight of
    any minimum spanning tree of a graph G is unique, even when multiple
    MSTs exist (due to weight ties).  Both Prim's (cut-property greedy)
    and Kruskal's (sort-and-union-find greedy) are proven correct, so
    w(Prim-MST) = w(Kruskal-MST) = w*(G), where w* is the optimal MST
    weight.

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes and strictly
    positive weights drawn from [0.01, 100.0].  Strictly positive weights
    make exact weight ties rare, further strengthening the comparison
    because the MST is likely unique edge-by-edge.

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected weighted graph with strictly positive weights.

    How a failure indicates a bug
    ─────────────────────────────
    A weight discrepancy proves that at least one of the two algorithms
    returned a spanning tree whose total weight is not minimal.  Since
    Kruskal's algorithm is used as a reference oracle, a failure most
    likely points to a bug in Prim's implementation (e.g., incorrect
    priority-queue ordering, wrong key-decrease logic, or edge-weight
    corruption during tree construction).

    Reference graph
    ───────────────
    See ``reference_graphs/test_prim_weight_equals_kruskal.png``
    """
    prim_mst = nx.minimum_spanning_tree(G, algorithm="prim")
    kruskal_mst = nx.minimum_spanning_tree(G, algorithm="kruskal")

    prim_w = sum(d["weight"] for _, _, d in prim_mst.edges(data=True))
    kruskal_w = sum(d["weight"] for _, _, d in kruskal_mst.edges(data=True))

    assert math.isclose(prim_w, kruskal_w, rel_tol=1e-6), (
        f"Prim weight {prim_w:.6f} ≠ Kruskal weight {kruskal_w:.6f}"
    )


# =============================================================================
#  DIJKSTRA'S  SHORTEST  PATH
#  Property-Based Tests
# =============================================================================


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_path_validity(data):
    """
    Property
    ────────
    The path returned by Dijkstra's algorithm is a valid path in G: it
    starts at the source, ends at the target, and every consecutive pair
    of nodes along the path is connected by an edge in G.

    Property type: Postcondition / output validity.

    Why this property matters
    ─────────────────────────
    Path validity is the most fundamental postcondition of any
    path-finding algorithm.  If the returned sequence of nodes does not
    correspond to actual edges in the graph, the "path" cannot be
    traversed and is meaningless.  This test verifies the structural
    integrity of the output before any optimality claim is even relevant.

    Mathematical reasoning
    ──────────────────────
    A path P = (v₀, v₁, …, vₖ) in graph G is valid if and only if:
      (a) v₀ = source,
      (b) vₖ = target, and
      (c) ∀ i ∈ {0, …, k−1}: (vᵢ, vᵢ₊₁) ∈ E(G).
    Dijkstra's algorithm constructs paths by following predecessor
    pointers set during edge relaxation; each relaxation uses an existing
    edge, so every step in the reconstructed path should correspond to
    an edge of G.

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes.  Both sparse
    and dense graphs are generated, and source/target pairs are drawn
    independently (they may be equal, which is a valid boundary case
    producing a single-node path).

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected graph with strictly positive edge weights.
    • Source and target are valid nodes in G (guaranteed by the strategy).

    How a failure indicates a bug
    ─────────────────────────────
    • path[0] ≠ source or path[-1] ≠ target → the algorithm returned a
      path for the wrong node pair, indicating a bug in predecessor
      tracking or path reconstruction.
    • A missing edge along the path → the algorithm followed a
      non-existent edge during relaxation, indicating corrupted adjacency
      data or an incorrect predecessor pointer.

    Reference graph
    ───────────────
    See ``reference_graphs/test_dijkstra_path_validity.png``
    """
    G, u, v = data
    path = nx.dijkstra_path(G, u, v, weight="weight")

    assert path[0] == u, (
        f"Path must start at source {u}, but starts at {path[0]}"
    )
    assert path[-1] == v, (
        f"Path must end at target {v}, but ends at {path[-1]}"
    )
    for i in range(len(path) - 1):
        assert G.has_edge(path[i], path[i + 1]), (
            f"Edge ({path[i]}, {path[i + 1]}) on the returned path "
            f"does not exist in G"
        )


@given(G=connected_weighted_undirected_graph())
def test_dijkstra_self_distance_is_zero(G):
    """
    Property
    ────────
    The shortest-path distance from any node to itself is exactly 0.

    Property type: Boundary condition / base-case invariant.

    Why this property matters
    ─────────────────────────
    The self-distance of zero is the base case on which Dijkstra's entire
    relaxation process is built.  The algorithm initialises dist[source]
    = 0 and relaxes outward.  If this initialisation is wrong, every
    subsequent distance computed from that source will be incorrect.

    Mathematical reasoning
    ──────────────────────
    The empty path (zero hops, staying at the source) has total weight 0.
    With all edge weights strictly positive, traversing any cycle and
    returning to the source would add positive cost, so no alternate path
    from a node to itself can achieve cost < 0.  Therefore
    dist(v, v) = 0 for every vertex v in a graph with non-negative
    weights.

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes and strictly
    positive edge weights.  The test iterates over *every* node in each
    generated graph, so boundary cases (leaf nodes, high-degree hubs) are
    all covered.

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected graph with non-negative edge weights (our
      strategy uses strictly positive weights, which is stronger).

    How a failure indicates a bug
    ─────────────────────────────
    A non-zero self-distance indicates a corrupted distance
    initialisation — for example, the algorithm may be initialising
    dist[source] to infinity instead of 0, or it may be traversing a
    negative-weight cycle that should not exist.  Either case represents
    a fundamental correctness bug in the distance-initialisation or
    relaxation logic.

    Reference graph
    ───────────────
    See ``reference_graphs/test_dijkstra_self_distance_is_zero.png``
    """
    for node in G.nodes():
        dist = nx.dijkstra_path_length(G, node, node, weight="weight")
        assert dist == 0, (
            f"Self-distance for node {node} is {dist}, expected 0"
        )


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_symmetry(data):
    """
    Property
    ────────
    On an undirected graph, dist(u, v) == dist(v, u).

    Property type: Invariant / metric-space axiom.

    Why this property matters
    ─────────────────────────
    In an undirected graph, every edge (u, v, w) is traversable in both
    directions at equal cost w.  Shortest-path distances on such graphs
    form a metric space, and symmetry is one of the three metric axioms
    (along with non-negativity and the triangle inequality).  A
    shortest-path algorithm that violates symmetry on an undirected graph
    has a directional bias that should not exist.

    Mathematical reasoning
    ──────────────────────
    Let P = (u = x₀, x₁, …, xₖ = v) be a shortest path from u to v
    with total weight W.  Because G is undirected, the reversed path
    P' = (v = xₖ, xₖ₋₁, …, x₀ = u) traverses the same edges in
    reverse order.  Each edge has the same weight in both directions, so
    cost(P') = cost(P) = W.  Since P is optimal from u to v, P' provides
    a path from v to u of cost W, implying dist(v, u) ≤ W = dist(u, v).
    By the symmetric argument, dist(u, v) ≤ dist(v, u).  Therefore
    dist(u, v) = dist(v, u).

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes and strictly
    positive weights.  Two distinct nodes are sampled per test case
    (enforced by ``assume(u != v)``).

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected graph; u ≠ v.

    How a failure indicates a bug
    ─────────────────────────────
    Asymmetry implies the implementation is treating the undirected graph
    as directed — for example, by using a directed adjacency structure
    internally, or by applying direction-sensitive edge-weight logic.
    This is a bug in graph traversal: the algorithm processed edges in
    only one direction when it should have processed both.

    Reference graph
    ───────────────
    See ``reference_graphs/test_dijkstra_symmetry.png``
    """
    G, u, v = data
    assume(u != v)

    dist_uv = nx.dijkstra_path_length(G, u, v, weight="weight")
    dist_vu = nx.dijkstra_path_length(G, v, u, weight="weight")

    assert math.isclose(dist_uv, dist_vu, rel_tol=1e-9), (
        f"Symmetry violated: dist({u}, {v}) = {dist_uv:.9f} "
        f"≠ dist({v}, {u}) = {dist_vu:.9f}"
    )


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_triangle_inequality(data):
    """
    Property
    ────────
    For all nodes u, v, w in a connected graph:
        dist(u, v) ≤ dist(u, w) + dist(w, v)   (triangle inequality).

    Property type: Invariant / metric-space axiom.

    Why this property matters
    ─────────────────────────
    The triangle inequality is a direct consequence of shortest-path
    optimality.  It states that the direct shortest path between two
    nodes can never be more expensive than any detour through a third
    node.  Violating this inequality means the algorithm failed to find
    a cheaper path that provably exists, which is a correctness failure.

    Mathematical reasoning
    ──────────────────────
    Proof by contradiction: suppose dist(u, v) > dist(u, w) + dist(w, v)
    for some intermediate node w.  Then concatenating the optimal path
    u → w with the optimal path w → v yields a valid u-to-v path with
    total length dist(u, w) + dist(w, v) < dist(u, v), contradicting
    the claim that dist(u, v) is the shortest-path distance.  Therefore
    the inequality must hold for every correct shortest-path algorithm.

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes.  For a sampled
    pair (u, v), the test iterates over *every* third node w in the
    graph, providing exhaustive coverage of all possible intermediate
    detours.  A small absolute tolerance (1e-9) accounts for
    floating-point summation-order differences.

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected graph with non-negative edge weights.

    How a failure indicates a bug
    ─────────────────────────────
    A violation means the algorithm computed an incorrect shortest-path
    distance for at least one node pair: a cheaper route through an
    intermediate node w exists but was not discovered.  This points to
    a bug in the relaxation step (failing to update a shorter estimate),
    a corrupted priority queue (processing nodes in the wrong order), or
    premature termination (stopping before all reachable nodes are
    finalised).

    Reference graph
    ───────────────
    See ``reference_graphs/test_dijkstra_triangle_inequality.png``
    """
    G, u, v = data
    dist_uv = nx.dijkstra_path_length(G, u, v, weight="weight")

    for w in G.nodes():
        dist_uw = nx.dijkstra_path_length(G, u, w, weight="weight")
        dist_wv = nx.dijkstra_path_length(G, w, v, weight="weight")
        assert dist_uv <= dist_uw + dist_wv + 1e-9, (
            f"Triangle inequality violated: "
            f"dist({u}, {v}) = {dist_uv:.9f}  >  "
            f"dist({u}, {w}) + dist({w}, {v}) = "
            f"{dist_uw:.9f} + {dist_wv:.9f}"
        )


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_subpath_optimality(data):
    """
    Property
    ────────
    Every prefix of a shortest path from u to v is itself a shortest
    path from u to the prefix's terminal node (optimal substructure /
    Bellman principle of optimality).

    Property type: Metamorphic / optimal-substructure invariant.

    Why this property matters
    ─────────────────────────
    Dijkstra's correctness is built on the principle of optimal
    substructure: if the overall path is optimal, then every initial
    segment must also be optimal.  This property is the theoretical
    foundation that allows greedy algorithms like Dijkstra to guarantee
    a global optimum by making locally optimal choices.  Violating it
    would mean the algorithm is internally inconsistent.

    Mathematical reasoning
    ──────────────────────
    Let P[u→v] = (u = x₀, x₁, …, xₖ = v) be an optimal path.  For any
    index j < k, consider the prefix P[u→xⱼ] = (x₀, …, xⱼ).

    Proof by contradiction: suppose there exists a cheaper path Q from u
    to xⱼ, i.e., cost(Q) < cost(P[u→xⱼ]).  Then the concatenation
    Q ⊕ P[xⱼ→v] is a valid u-to-v path with total cost:
        cost(Q) + cost(P[xⱼ→v])
        < cost(P[u→xⱼ]) + cost(P[xⱼ→v])
        = cost(P[u→v])
    This contradicts the optimality of P[u→v].  Therefore P[u→xⱼ] must
    itself be optimal from u to xⱼ.  ∎

    Graphs tested
    ─────────────
    Connected undirected weighted graphs with 2–15 nodes.  Two distinct
    nodes are sampled (enforced by ``assume(u != v)``), and every prefix
    of the shortest path is checked — not just a random sub-segment.
    This provides exhaustive coverage of all intermediate waypoints.

    Assumptions / preconditions
    ───────────────────────────
    • Connected undirected graph with strictly positive edge weights.
    • u ≠ v so the path has at least one edge.

    How a failure indicates a bug
    ─────────────────────────────
    A discrepancy between the prefix weight and Dijkstra's independently
    reported optimal distance to that prefix endpoint means the algorithm
    is internally inconsistent: it returned a path that is locally
    suboptimal at some intermediate step.  This points to a bug in the
    predecessor-tracking or path-reconstruction logic — the algorithm
    found a shorter route to an intermediate node but failed to
    incorporate it into the final path, or vice versa.

    Reference graph
    ───────────────
    See ``reference_graphs/test_dijkstra_subpath_optimality.png``
    """
    G, u, v = data
    assume(u != v)

    path = nx.dijkstra_path(G, u, v, weight="weight")

    for end_idx in range(1, len(path)):
        prefix = path[: end_idx + 1]
        prefix_weight = sum(
            G[prefix[i]][prefix[i + 1]]["weight"]
            for i in range(len(prefix) - 1)
        )
        optimal_dist = nx.dijkstra_path_length(
            G, u, prefix[-1], weight="weight"
        )
        assert math.isclose(
            prefix_weight, optimal_dist, rel_tol=1e-9, abs_tol=1e-9
        ), (
            f"Subpath optimality violated: prefix u={u} → {prefix[-1]} "
            f"via path has weight {prefix_weight:.9f} but Dijkstra reports "
            f"optimal distance {optimal_dist:.9f}"
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
