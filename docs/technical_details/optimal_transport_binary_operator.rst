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
A simple capacity constraint is

.. math::

    \sum_a m^X_a r^X_{a,p}
    + \sum_b m^Y_b r^Y_{b,p}
    + \sum_t m^Z_t z_{t,p}
    \le M_p.

This is a static residency bound. A temporally scheduled implementation would
replace it with one constraint per execution interval.

Objective
+++++++++

Let :math:`s_X(a)` and :math:`s_Y(b)` denote the initial storage sites, and let
:math:`c_{u,v}` be the measured or estimated cost per byte transferred from
site :math:`u` to site :math:`v`. The transport cost is

.. math::

    C_{\mathrm{move}} =
      \sum_{a,p} c_{s_X(a),p} m^X_a r^X_{a,p}
      + \sum_{b,p} c_{s_Y(b),p} m^Y_b r^Y_{b,p}
      + \sum_{t,p} c_{p,d(t)} m^Z_t w_{t,p}.

Pure byte minimization can assign all work to one site. To represent execution
time, introduce a makespan bound :math:`H` and per-tile compute estimates
:math:`\tau_{t,p}`:

.. math::

    \sum_t \tau_{t,p} z_{t,p} \le H
    \qquad \forall p \in P.

The complete first objective is

.. math::

    \min \quad C_{\mathrm{move}} + \lambda H,

where :math:`\lambda` converts execution time to the same optimization scale
as movement cost. Further terms may represent packing, conversion, contention,
or eviction, but each term must correspond to a measurable implementation
cost.

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
