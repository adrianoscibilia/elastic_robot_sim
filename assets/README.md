# Robot assets

This directory contains robot descriptions and small, redistributable benchmark
artifacts needed to reproduce the simulator integrations.  Large raw datasets
are intentionally excluded from version control; each asset-specific
`PROVENANCE.md` records its upstream location, licence, checksum and retrieval
command.

Robot-specific configuration belongs next to the corresponding asset, while
shared loaders/builders must not depend on a particular robot directory.
