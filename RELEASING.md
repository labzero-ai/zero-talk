# Releasing

The `release.yml` workflow publishes Python, npm and Rust packages from a version tag.
Each publishing job waits for the builds and checks, then requires approval through
the GitHub environment `release`. No registry token is stored in the repository or
used on a local machine by this workflow.

## One-time owner setup

- Create the npm organisation `labzero-ai` and give the lab account permission to
  publish `@labzero-ai/zero-talk` publicly.
- Create the GitHub environment **release**, add required reviewers and restrict
  deployments to release tags (`v*`). See [GitHub environment protection](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
- Configure each registry's GitHub trusted publisher with owner **labzero-ai**,
  repository **zero-talk**, workflow filename **release.yml** and environment **release**:
  - **PyPI, `zero-talk`:** use a [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
    for a new project. No manual first upload is needed. The workflow enables
    [PyPI attestations](https://docs.pypi.org/attestations/producing-attestations/).
  - **npm, `@labzero-ai/zero-talk`:** configure it in the existing package's
    Settings → Trusted publishing, and allow direct **npm publish**. The
    [npm guide](https://docs.npmjs.com/trusted-publishers/) requires npm 11.5.1+
    and Node 22.14.0+ on a hosted runner; this workflow installs npm 11 on Node 24.
    It also requires `repository.url` in `js/package.json` to identify
    `https://github.com/labzero-ai/zero-talk`. Complete the first trusted publish
    within two days of configuration, or recreate the expired publisher.
  - **crates.io, both `zero-talk-model-11m` and `zero-talk`:** configure each crate
    separately. [crates.io explicitly requires an existing published crate](https://crates.io/docs/trusted-publishing).

**Initial npm and crates.io publication is a separate owner decision.** npm's
documented setup starts in an existing package's settings and offers no pending
publisher; a first non-OIDC publication is therefore needed before that setup.
This is an inference from the [npm setup instructions](https://docs.npmjs.com/trusted-publishers/),
also tracked in the [npm documentation issue](https://github.com/npm/documentation/issues/1926).
crates.io explicitly requires an API token for the initial publication. Arrange
these bootstrap releases separately using CI-only credentials, then revoke them;
this workflow has no token fallback. Choose initial versions separately: a version
already published during bootstrap cannot be published again by this workflow.

The release builds only from this public repository. A build adds one new kind of data,
the runner's folders: the
workflow remaps them, and `.github/scripts/check-artifacts.py` fails the run if one
reaches an archive or a binary.

## Release steps

1. Set the same version in `rust/Cargo.toml` (`workspace.package`),
   `python/pyproject.toml`, `python/Cargo.toml` and `js/package.json`. Update internal
   crate dependency versions, lockfiles and version assertions where needed.
   Ensure the two Rust crates permit publication and the npm repository metadata
   matches the public repository.
2. Review the changes and pass CI. It includes crate packaging and an npm pack dry
   run. Commit the reviewed changes.
3. Create `vVERSION` on that commit and push the tag.
   For version 1.0.0, the tag is `v1.0.0`. Never move a released tag.
4. Review the release build results and approve the three `release` environment
   jobs. Verify the versions and files in all three registries afterward.

Wheels are built and tested natively on Linux x86-64 and ARM64 (manylinux 2.28),
Windows x86-64, macOS Apple silicon (`macos-26`) and macOS Intel (`macos-26-intel`).
The macOS labels are [GitHub-hosted runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
A source distribution
is also built. PyPI attestations and npm provenance are enabled.

For a manual run, select the existing version tag in **Actions → release → Run
workflow**. A branch run fails the version gate. Registries publish independently;
there is no rollback across them. After a partial release, inspect the registries
and rerun only failed jobs; rerunning successful publishers tries to republish an
immutable version.
