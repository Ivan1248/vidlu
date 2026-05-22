# Agent Instructions

Version: 2026-04-20

These are the user rules that all coding agents should follow in this repository.

See [GOALS.md](GOALS.md) for current high-level project goals and progress.

See [docs/conventions.md](docs/conventions.md) for code style and other conventions.

## Working style

- Make sure that you understand the purpose of what you are doing. It is important for catching misunderstandings and my mistakes.
- if you can't find or don't know something, don't assume. Ask me to clarify.

## Hypothesis testing / experiments

- Before testing any hypothesis, state:
  - what you expect,
  - and what the correct conclusions would be depending on the outcome.
- Update based on the observations.

## Units and dates

- Prefer SI units.
- Prefer the YYYY-MM-DD date format.

## Notes

- For markdown files used for development, use the `.devdocs/` directory.
- Use `/.devdocs/context.md` for noting/finding important observations (surprises, insights, agent-relevant discoveries).
- Be wary of too many high-level verification claims; if important, include probability of correctness in parentheses.
- Keep notes correct, up to date, and relevant.
- See also [.agents/stack-overflow-notes.md](.agents/stack-overflow-notes.md) for instructions about writing "Stack Overflow notes".

## Code quality and design

- Write correct code that avoids silent errors and unjustified fallbacks.
- Prefer the simplest design that meets the requirements; abstract when it removes real duplication or clarifies the code.
- Factor out shared logic.
- Prefer pure functions and immutable data where practical; keep I/O and shared-state mutation at the edges.
- Don't mutate arguments or aliased data unless that's the function's documented purpose.
- Separate distinct concerns; split units that mix responsibilities.
- Keep a single source of truth for values that must stay consistent.
- A function's parameters should reflect what it actually uses: no hidden dependencies on instance state or closures, and no broader objects than necessary.

## Naming

- Names that are not used very locally should be descriptive and sufficiently specific, not generic.
- If you simplify a function `foo`, keep the name `foo` rather than renaming it to something like `simplified_foo` / `simplifiedFoo` when no other `foo` remains.
- Prefix count variables with `num` rather than suffixing with `count` (follow the repository's existing case style).

## API changes and deletions

- You may break APIs freely to improve design, naming, and structure.
- Don’t worry about backward compatibility and migration unless explicitly requested.