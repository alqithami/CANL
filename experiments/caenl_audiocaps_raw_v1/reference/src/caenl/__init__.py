"""CAENL Franklin Open revision pipeline.

Standalone, file-logged, resumable experiment pipeline for the major revision of
"CAENL: Collapse-Based Active Neural Learning" (FRAOPE-D-25-02381).

Every job writes ``job.json`` (resolved config), ``events.jsonl``, ``metrics.jsonl``,
``system.csv``, ``summary.json`` and artifacts into its own directory.  Tables and
figures are produced exclusively from those records by ``caenl report``.
"""

__version__ = "5.4.3"
