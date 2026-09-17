Testing
=======

The test suite is organized in tiers. Unit tests run everywhere with no
services; the integration tiers need the Docker Compose stack.

.. mermaid::

    flowchart TB
        subgraph tiers[Test tiers]
            direction TB
            U["<b>Unit</b><br/>pytest -m 'not integration'<br/>mocked clients, fast"]
            I["<b>Integration</b><br/>live ES / OpenSearch / MongoDB"]
            F["<b>Fault injection</b><br/>real TCP faults via Toxiproxy"]
            M["<b>Mutation</b><br/>mutmut, weekly + on demand"]
            U --> I --> F
            M -.->|hardens| U
        end

Unit tests
----------

.. code-block:: bash

    make test            # pytest -m 'not integration'

Covers signature generation, record/word encoding, driver logic with mocked
backend clients, and negative paths — backend connection/write/delete
failures, malformed responses, worker and cursor errors, invalid inputs, and
migration failure modes (``tests/test_faults_unit.py``).

Integration tests
-----------------

.. code-block:: bash

    make test-integration-local   # starts services, runs tests, stops them

Runs against live Elasticsearch, OpenSearch 2.x/3.x and MongoDB. This
includes the fault-injection tier (``tests/test_faults_integration.py``),
which routes a driver through a `Toxiproxy <https://github.com/Shopify/toxiproxy>`_
proxy and injects real transport faults — proxy kill, latency, connection
reset — that the mock layer cannot reproduce:

.. mermaid::

    flowchart LR
        T["pytest"] -->|search / insert| D["image-match driver"]
        D -->|"HTTP :22300"| P["Toxiproxy proxy"]
        P --> OS["OpenSearch :9201"]
        A["Toxiproxy API :8475"] -.->|"toxics:<br/>latency, reset_peer, down"| P

``make db-up`` starts toxiproxy alongside the databases (API on ``:8475``,
dynamic proxy listeners on ``:22300-22310``).

Mutation testing
----------------

`mutmut <https://mutmut.readthedocs.io/>`_ generates mutants of
``src/image_match/`` and ``tools/migrate_to_knn.py`` and runs the unit suite
against each. It is deliberately not part of PR validation — a full run takes
a while — so it runs weekly and on demand via the ``mutation`` workflow:

.. code-block:: bash

    make mutation        # mutmut run + results

``mutmut run`` is incremental: re-running only re-tests mutants in functions
whose source changed. Lines that can never change behaviour (CLI help text,
``print`` reporting, the vestigial ``handle_mpo`` parameter, ``typing.cast``)
are excluded via ``do_not_mutate_patterns`` in ``pyproject.toml``.
