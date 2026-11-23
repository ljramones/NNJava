# Repository Guidelines

## Project Structure & Module Organization

This repository is an initial scaffold: `README.md` contains a placeholder, and no source code, tests, assets, or build configuration are present yet. Update the README as functionality is introduced. For new implementation, use clear directories such as `src/`, `tests/`, and `assets/` where appropriate; create them only when needed. Keep related code together and document the chosen layout.

## Build, Test, and Development Commands

No build, run, or test commands are configured. When adding a toolchain, document prerequisites and exact commands in `README.md` alongside the configuration files.

Current repository checks:

- `git status --short`: inspect pending changes before committing.
- `git diff --check`: detect whitespace errors in tracked changes.
- `git diff --stat`: review the scope of tracked changes.

These checks do not build or test application behavior. Review newly created files separately until they are staged.

## Coding Style & Naming Conventions

No language-specific style, formatter, or linter has been established. Match existing conventions once implementation begins, and adopt the chosen language's standard tooling through checked-in configuration. Use consistent indentation within each file, descriptive identifiers, and filenames that reflect purpose. For Markdown, use descriptive headings, fenced code blocks for multiline examples, and concise instructions.

## Testing Guidelines

No test framework or coverage requirement currently exists. Introduce automated tests with the first executable functionality, document how to run them, and follow the selected framework's test naming conventions. Cover expected behavior, boundary cases, and failure paths. Include a regression test when fixing a reproducible defect.

## Commit & Pull Request Guidelines

Git history currently contains only `initial`, so no established commit convention can be inferred. Use short, imperative subjects, such as `Add training data loader`, and keep commits focused.

Pull requests should describe the change, explain its purpose, link relevant issues, and report validation performed. State explicitly when tests cannot run or tooling is not yet configured. Update documentation whenever setup, commands, or repository structure changes.
