# Sources and adaptations

This workspace includes the original ActQuant tracked source under src/ActQuant, pinned and verified by its .upstream-snapshot.json. Its upstream license is preserved. The source remains unchanged; LeRobot input adaptation, diagnostics, reproducible calibration and publication tools live outside that directory. OpenPI and LeRobot are pinned dependencies listed in UPSTREAM_SOURCES.json.

The quantized artifact is derived from lerobot/pi05_libero_finetuned_v044 revision 7152f13e34c6c115d808aea0e5d08aca74eac14b. Tokenizer provenance is google/paligemma-3b-pt-224 revision 35e4f46485b4d07967e7e9935bc3786aad50687c. Consult the upstream model repositories for model terms; the source-code LICENSE alone does not describe model provenance.

Operational handoff history, credentials, environments, original weights and datasets are not part of this repository. Transfer-script server defaults are replaced with placeholders. Runtime and statistical limitations are documented with the model results.
