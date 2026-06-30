# Corpora

Reproducible binary corpora for the RQ1 token-savings study. Each builder emits
binaries plus a `corpus_manifest.json` that `run_study --manifest` reads to tag
every record with `{corpus, version, opt_level, stripped, arch, sha256}`.

The corpus is built at **`-O2`**, the realistic *and* canonical configuration:
GNU **Autoconf** sets `CFLAGS="-g -O2"` for GCC unless overridden, so `-O2` is
coreutils' own default build
([manual](https://www.gnu.org/software/autoconf/manual/autoconf-2.69/html_node/C-Compiler.html)),
and it matches the binaries an analyst actually meets.

## `coreutils.sh`: the paper's corpus

Fetches a pinned coreutils release, builds it (letting the autotools default
`-g -O2` stand), strips the binaries, and writes a manifest. Ground-truth source
is available for sanity-checking, and the binaries are redistributable. This is
the corpus behind the paper's RQ1 results (GNU coreutils 9.5).

```bash
evaluation/corpora/coreutils.sh /path/to/coreutils_bin
COREUTILS_VERSION=9.5 evaluation/corpora/coreutils.sh ./coreutils_bin
```

> **ISA:** binaries are native to the build host (Mach-O arm64 on macOS,
> ELF x86-64 on a Linux x86-64 box/container). For the paper's canonical ELF
> x86-64 target, run the script in a Linux x86-64 environment (e.g. a Docker
> container).

## Running the study with provenance

```bash
# manifest auto-detected from the bin dir
python -m evaluation.run_study --bin-dir coreutils_bin decompiled/ study.json
```

Each `files[]` record gains a `provenance` block tagging it with
`{corpus, version, opt_level, stripped, arch, sha256}` from the manifest.
