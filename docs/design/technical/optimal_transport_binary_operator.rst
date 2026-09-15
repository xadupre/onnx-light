.. _l-technical-details-optimal-transport-binary-operator:

Optimal transport formulation for a binary operator
====================================================

This page formalizes the implementation of a binary tensor operator as a
data-placement problem. The operator semantics remain fixed; an optimizer only
chooses where output tiles are computed and which operand tiles are transferred
or replicated at each execution site.

The first case is an elementwise ``Add`` with multidirectional broadcasting.
It is deliberately simpler than ``MatMul``: every output element has one
independent binary computation, so the transport plan cannot change floating
point accumulation order.

Scope
-----

Let

.. math::

    Z = X \mathbin{\oplus} Y

where :math:`\oplus` is a deterministic elementwise binary operation and
``X`` and ``Y`` broadcast to the output shape ``S``. The formulation applies
directly to operators such as ``Add``, ``Sub``, ``Mul``, ``Div``, ``Max``, and
logical comparisons when their output elements are independent.

The model chooses:

* an execution site for every output tile;
* which input tiles must be present at each selected site;
* where each output tile is stored after computation.

It does not choose:

* the mathematical result of the operator;
* a different broadcasting rule;
* an approximation or a different numerical order;
* runtime thread scheduling inside one execution site.

Why classical two-marginal transport is insufficient
------------------------------------------------------

Classical optimal transport moves one conserved mass distribution onto another.
A binary computation is different: every output tile jointly requires mass
from two operands, and broadcasting may replicate one input element many
times. Consequently, the implementation problem is not a plain Wasserstein
distance between ``X`` and ``Y``.

The appropriate discrete model is a **replicated, capacitated,
multi-commodity transport problem**:

* ``X`` tiles are one commodity;
* ``Y`` tiles are a second commodity;
* output tiles are computation demands;
* a demand is feasible only when both required commodities meet at its
  execution site;
* transported input tiles may be reused by several demands at that site;
* broadcasting permits replication and therefore replaces strict global mass
  conservation with explicit materialization decisions.

This distinction is important. Calling the problem "optimal transport" is
useful only if replication, co-location, capacity, and computation are stated
as constraints rather than hidden inside an informal cost.

Broadcast semantics
-------------------

Let the output iteration domain be

.. math::

    \Omega = \prod_{d=0}^{r-1} \{0,\ldots,S_d-1\}.

After left-padding operand shapes with ones to rank :math:`r`, broadcasting
defines two fixed projections:

.. math::

    \pi_X : \Omega \rightarrow \Omega_X,\qquad
    \pi_Y : \Omega \rightarrow \Omega_Y.

For output index :math:`i \in \Omega`, projection component :math:`d` is

.. math::

    \pi_X(i)_d =
    \begin{cases}
      0   & \text{if } X_d = 1,\\
      i_d & \text{otherwise,}
    \end{cases}

and likewise for :math:`\pi_Y`. The computation is therefore fixed:

.. math::

    Z_i = X_{\pi_X(i)} \mathbin{\oplus} Y_{\pi_Y(i)}.

No optimization variable appears in this equation. The transport model starts
only after the semantic projections have been established.

Tiled problem
-------------

The output domain is partitioned into computation tiles :math:`t \in T`.
The two projected input domains are partitioned into operand tiles
:math:`a \in A` and :math:`b \in B`.

Define the incidence constants

.. math::

    q^X_{a,t} =
    \begin{cases}
      1 & \text{if output tile } t \text{ reads input tile } a,\\
      0 & \text{otherwise,}
    \end{cases}

and :math:`q^Y_{b,t}` similarly. These constants are derived exactly from
:math:`\pi_X` and :math:`\pi_Y`; they are not estimated by the optimizer.

Let :math:`P` be the set of execution sites. A site may represent a NUMA node,
a cache-sharing core group, a worker-local packed buffer, or another level at
which data residency is meaningful.

Memory-hierarchy transport network
++++++++++++++++++++++++++++++++++

An execution site alone is too coarse to predict latency. For a CPU, expand it
into a directed graph :math:`G=(V,E)`. A typical input path is

.. math::

    \text{RAM} \rightarrow \text{L3} \rightarrow \text{L2}
    \rightarrow \text{L1} \rightarrow \text{register}
    \rightarrow \text{ALU},

and an output follows the reverse path. Shared caches and links appear only
once in the graph, so traffic from different cores competes for the same
resources. Separate directed edges allow read and write bandwidths to differ.

For every hierarchy edge :math:`e`, define:

* :math:`\ell_e`, the transfer granularity in bytes, such as a cache line;
* :math:`q_e`, the maximum payload of one transfer transaction in bytes;
* :math:`n_e`, the maximum number of transactions in flight;
* :math:`B_e`, the sustainable bandwidth in bytes per second;
* :math:`L_e`, the unloaded latency of one transaction.

For a scalar type of size :math:`b` bytes, one transaction transports at most

.. math::

    N^{\mathrm{transaction}}_e =
    \left\lfloor \frac{q_e}{b} \right\rfloor

elements, while at most

.. math::

    N^{\mathrm{flight}}_e =
    n_e N^{\mathrm{transaction}}_e

elements are simultaneously in flight on the edge. Its ideal sustained scalar
rate is :math:`B_e/b` elements per second. These are different limits: payload
answers "how many at once", whereas bandwidth answers "how many per second".
Measured :math:`B_e` already includes protocol overhead; otherwise only the
payload fraction of each transaction may be counted.

Each storage node :math:`v` has a usable residency capacity :math:`K_v` in
bytes. In particular, :math:`K_{\mathrm{L1}}`,
:math:`K_{\mathrm{L2}}`, and :math:`K_{\mathrm{L3}}` constrain the live input,
output, and temporary tiles at those levels. A cache can hold at most

.. math::

    N^{\mathrm{resident}}_v =
    \left\lfloor \frac{K_v}{b} \right\rfloor

scalars of size :math:`b` when no other data occupies it. A binary operation
with two same-sized inputs and one output has a smaller ideal live working-set
bound, :math:`\lfloor K_v/(3b) \rfloor` output elements, before accounting for
cache lines, associativity, code, and unrelated data.

An ALU node additionally has a service rate :math:`R_v` in scalar operations
per second, and a core has a finite number :math:`h_v` of hardware-thread
contexts. The abstraction may assign one output element to one software
thread, one thread to one context, and one context to one core, but it must
enforce :math:`h_v`; creating more software threads does not create more
simultaneous processors.

Contiguous elements are cheaper because a hierarchy normally transfers whole
cache lines. For a set :math:`U` of scalar addresses crossing edge :math:`e`,
the charged byte count is

.. math::

    D_e(U) =
    \ell_e
    \left|
      \left\{
        \left\lfloor \frac{\operatorname{addr}(u)}{\ell_e} \right\rfloor
        : u \in U
      \right\}
    \right|.

Adjacent elements in one line therefore share a transaction. Strided elements
may each require a different line even when the useful scalar byte count is
the same. This definition also prevents broadcast reuse from being charged
again while the line remains resident; an eviction followed by a reload is a
new transfer.

For a simultaneous execution wave of duration :math:`\Delta`, let
:math:`F_{e,k}` be the bytes sent over edge :math:`e` during interval
:math:`k`. Feasibility requires

.. math::

    F_{e,k} \le B_e \Delta
    \qquad \forall e,k,

and the number of outstanding transactions cannot exceed :math:`n_e`.
Residency at every instant similarly requires

.. math::

    \sum_u \operatorname{bytes}(u) \rho_{u,v,k} \le K_v
    \qquad \forall v,k,

where :math:`\rho_{u,v,k}` states that tile or cache line :math:`u` is resident
at node :math:`v` during interval :math:`k`. These time-indexed constraints
make contention explicit: calculations may start together, but transfers that
share a saturated L3 or RAM link cannot all complete at their unloaded rate.

For a single stream whose path is :math:`P_{\mathrm{mem}}`, the steady-state
upper bound is the bottleneck

.. math::

    R_{\mathrm{stream}} \le
    \min\left(
      R_{\mathrm{ALU}},
      \min_{e \in P_{\mathrm{mem}}}
      \frac{B_e}{d_e}
    \right),

where :math:`d_e` is the charged number of bytes crossing edge :math:`e` per
output element, including both operands and the result. For example,
:math:`d_e=3b` only when two input scalars and one output scalar really cross
that edge for every operation; cache reuse and write policies change it.
Unloaded end-to-end latency is at least the sum of stage latencies on the
critical dependency path, while long streams approach the bottleneck rate
after the pipeline fills.

For a charged volume :math:`D_e`, the isolated time model for one hierarchy
stage is

.. math::

    T_e(D_e) =
    \mathbb{1}_{D_e>0} L_e + \frac{D_e}{B_e}.

The first term charges pipeline startup and the second charges sustained
transfer. When transactions cannot overlap, replace the first term by
:math:`\lceil D_e/q_e \rceil L_e`. When stages pipeline, their byte-transfer
times must not all be added: the bottleneck determines steady-state throughput,
and only non-overlapped latency on the critical path is added. The parameters
:math:`K_v`, :math:`q_e`, :math:`n_e`, :math:`B_e`, and :math:`L_e` are
hardware-specific and should come from cache topology plus aligned,
strided, read, write, and mixed read/write microbenchmarks.

Decision variables
++++++++++++++++++

The binary assignment variable

.. math::

    z_{t,p} \in \{0,1\}

is one when output tile :math:`t` is computed at site :math:`p`.

The materialization variables

.. math::

    r^X_{a,p}, r^Y_{b,p} \in \{0,1\}

state that operand tile :math:`a` or :math:`b` is available at site
:math:`p`. One materialization may serve every computation tile assigned to
the same site, which is how the model represents cache or packed-buffer reuse.

Finally,

.. math::

    w_{t,p} \in \{0,1\}

states that the result of output tile :math:`t`, after being computed at
:math:`p`, must be transferred to its required destination.

Feasibility constraints
+++++++++++++++++++++++

Every output tile is computed exactly once:

.. math::

    \sum_{p \in P} z_{t,p} = 1
    \qquad \forall t \in T.

Both operands must be present at the selected site:

.. math::

    r^X_{a,p} \ge q^X_{a,t} z_{t,p}
    \qquad \forall a,t,p,

.. math::

    r^Y_{b,p} \ge q^Y_{b,t} z_{t,p}
    \qquad \forall b,t,p.

If :math:`d(t)` is the required destination of output tile :math:`t`, then

.. math::

    w_{t,p} \ge z_{t,p}
    \qquad \text{when } p \ne d(t).

Let :math:`m^X_a`, :math:`m^Y_b`, and :math:`m^Z_t` be tile sizes in bytes,
and let :math:`M_p` be the usable residency capacity at site :math:`p`.
When a site is not expanded into hierarchy nodes, a simple capacity constraint
is

.. math::

    \sum_a m^X_a r^X_{a,p}
    + \sum_b m^Y_b r^Y_{b,p}
    + \sum_t m^Z_t z_{t,p}
    \le M_p.

This is a static residency bound. The :math:`K_v` constraints above replace it
when the hierarchy and execution intervals are modeled explicitly.

Objective
+++++++++

Let :math:`s_X(a)` and :math:`s_Y(b)` denote the initial storage sites, and let
:math:`P(u,v)` be the hierarchy path from site :math:`u` to site :math:`v`.
For edge :math:`e`, a linear monetary or energy cost
:math:`c^{\mathrm{byte}}_e` may be charged per transferred byte and
:math:`c^{\mathrm{transaction}}_e` per transaction. Thus every step has both
the time cost :math:`T_e(D_e)` defined above and the optimization cost

.. math::

    C_{\mathrm{move}} =
    \sum_e \left(
      c^{\mathrm{byte}}_e D_e
      + c^{\mathrm{transaction}}_e
        \left\lceil \frac{D_e}{q_e} \right\rceil
    \right),

where :math:`D_e` is derived from selected materializations, cache-line
coverage, evictions, and result writes whose paths contain :math:`e`. This
edge sum replaces an opaque direct-site cost such as

.. math::

    c_{u,v}m =
    \sum_{e \in P(u,v)} c^{\mathrm{byte}}_e m,

which remains useful as a simplified model when topology details are unknown.
Costs on shared edges must be charged once to the aggregate cache-line traffic,
not independently to every element, or the model loses both spatial locality
and broadcast reuse.

Pure movement-cost minimization can assign all work to one site. To represent
execution time, introduce a makespan bound :math:`H`. It must be no smaller
than the work divided by the rate of every shared resource:

.. math::

    H \ge \frac{D_e}{B_e}
    \qquad \forall e \in E,

.. math::

    H \ge
    \frac{\sum_t o_t z_{t,p}}{R_p}
    \qquad \forall p \in P,

where :math:`o_t` is the scalar-operation count of tile :math:`t`. Dependency
paths add their non-overlapped transaction latencies :math:`L_e`; bandwidth
terms model the overlapped steady state. A time-indexed solver can determine
that overlap directly. A coarser solver may use the safe approximation

.. math::

    H \ge
    \max_e \frac{D_e}{B_e}
    + L_{\mathrm{critical}}
    + \max_p \frac{\sum_t o_t z_{t,p}}{R_p}.

The complete first objective is

.. math::

    \min \quad C_{\mathrm{move}} + \lambda H,

where :math:`\lambda` converts execution time to the same optimization scale
as movement cost. Capacity and throughput are constraints, not arbitrary
penalties. Further terms may represent packing, conversion, or energy, but
each term must correspond to a measurable implementation cost.

The operational integer model is a capacitated facility-location problem that
generalizes transport: assigning a task to a site opens the required operand
materializations there. Optimal transport supplies the movement vocabulary,
while facility location captures replication and reuse. The linear relaxation,
obtained by allowing variables in :math:`[0,1]`, is an analysis oracle; an
executable plan still requires deterministic integral rounding.

Concrete broadcast example
--------------------------

Consider

.. math::

    X \in \mathbb{R}^{2 \times 1},\qquad
    Y \in \mathbb{R}^{1 \times 4},\qquad
    Z = X + Y \in \mathbb{R}^{2 \times 4}.

There are two identical execution sites and every scalar has unit transfer
cost and unit size. Ignore output writes because all candidate plans write the
same eight output elements. Capacity is unbounded, so transport alone is
allowed to place all work at one site.

If all work is assigned to one site:

* the site receives both ``X`` scalars and all four ``Y`` scalars;
* input transport costs :math:`2 + 4 = 6` scalar transfers;
* the makespan is eight scalar operations.

If work is partitioned by rows:

* site 0 receives one ``X`` scalar and all four ``Y`` scalars;
* site 1 receives the other ``X`` scalar and all four ``Y`` scalars;
* input transport costs :math:`2 + 8 = 10` scalar transfers;
* both sites compute four outputs, so the makespan is four operations.

If two columns are assigned to each site:

* both sites receive both ``X`` scalars;
* every ``Y`` scalar is sent to exactly one site;
* input transport costs :math:`4 + 4 = 8` scalar transfers;
* both sites compute four outputs, so the makespan is four operations.

The objective values are therefore

.. math::

    C_{\mathrm{single}} = 6 + 8\lambda,\qquad
    C_{\mathrm{columns}} = 8 + 4\lambda,\qquad
    C_{\mathrm{rows}} = 10 + 4\lambda.

Transport alone selects the single-site plan. Once :math:`\lambda > 1/2`, the
balanced column partition wins; the row partition is dominated because it has
the same makespan and more movement. This example exposes both decisions
explicitly: the makespan term determines whether a second site is worth using,
then broadcasting determines which balanced partition replicates less data.

For two same-shaped operands without broadcasting, each balanced partition
normally transfers every required input element exactly once. The transport
term is then degenerate, and topology, packing, cache capacity, or makespan
must distinguish plans.

Relationship to an executable kernel
-------------------------------------

For an elementwise binary kernel, an integral solution maps directly to a
schedule:

1. Materialize each selected input tile according to :math:`r^X` and
   :math:`r^Y`.
2. Dispatch output tile :math:`t` to its unique site selected by :math:`z`.
3. Evaluate the existing scalar operation using the fixed broadcast
   projections.
4. Transfer or retain the result according to :math:`w`.

The solver does not belong in the elementwise hot path. Small instances can be
solved exactly offline to provide an oracle. Larger instances may use a linear
or entropy-regularized relaxation followed by deterministic rounding. The
resulting plan should be cached by shape, strides, element types, topology,
and available capacity.

Validation criteria
-------------------

A transport-derived implementation is useful only if all of the following
hold:

* it produces bit-identical output to the existing binary kernel;
* every output tile is covered exactly once;
* measured transferred bytes correlate with :math:`C_{\mathrm{move}}`;
* capacity constraints match peak temporary memory;
* the predicted makespan ordering correlates with measured latency;
* plan construction is outside the inference hot path;
* a fallback remains available when shapes or topology do not match the
  cached plan.

The same vocabulary can later be extended to reductions and matrix
multiplication, but those operators add accumulation dependencies, partial
results, and numerical-order constraints that are intentionally absent from
this first binary case.
