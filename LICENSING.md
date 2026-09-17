# Licensing notes for gollum — read before moving this into the new repo

Not part of the licence itself; a record of why it's shaped this way, so the
reasoning doesn't have to be reconstructed later.

## Why Apache-2.0, not MIT

- Explicit patent grant + a termination clause if a user sues over patents
  they hold — relevant because this wraps ML inference, a patent-dense area.
- A `NOTICE` file mechanism, used here for the llama-go/llama.cpp attribution.
- Recognised instantly by licence scanners (FOSSA/Snyk/Black Duck) and by
  `pkg.go.dev` — the whole point of publishing gollum is frictionless `go get`
  adoption, and an unrecognised custom licence defeats that for most
  potential users.

## Why not the Sustainable Use License (edith's licence)

SUL restricts distribution to free-of-charge, non-commercial use. A library
exists to be linked into other people's products; SUL would make every
commercial adopter's use a breach by definition. Correct for edith and
agippy (applications, not redistributed as dependencies); wrong for a
library. This was the reasoning that led here rather than just copying
edith's `LICENSE.md`.

## Why not PolyForm Noncommercial / the Caldun-style licence

Considered and rejected for gollum specifically: it gates on
commercial-vs-noncommercial use, which doesn't match the actual requirement.
A military R&D lab doing unfunded internal work is "noncommercial" under
that licence and would pass through unrestricted; a small paying commercial
customer would need to buy a separate licence for ordinary, unobjectionable
use. The real requirement is a field-of-use exclusion (military/weapons/mass
surveillance, full stop, not gated on payment), which neither PolyForm nor
SUL express — both gate on the wrong axis.

## Why the restriction is a separate section, not baked into Apache-2.0 text

Apache-2.0 is a permissive licence by definition; a field-of-use restriction
added to it is no longer OSI-approved "Apache-2.0" even if 95% of the text
is verbatim. That trade-off is accepted deliberately, following the same
precedent as:

- Meta's Llama licence — a source-available licence with an Acceptable Use
  Policy incorporated by reference, banning military/weapons/surveillance
  use unconditionally, no exception for payment.
- Anthropic's own Usage Policy — layered on top of product access rather
  than expressed as a copyright restriction, same effect.
- edith's own `LICENSE.md` "Additional Terms — Artificial Intelligence and
  Machine Learning" section, which already uses exactly this pattern
  (Sustainable Use License + a prevailing addendum in the same file).

**Known cost, not a defect to fix:** because of the addendum, GitHub's
licence detector and most SCA tooling will show this as "Other" / "custom"
rather than "Apache-2.0", and some corporate compliance processes will
require manual review before adopting it. That's the deliberate price of the
military exclusion — there's no licence text that gets both effortless
scanner recognition and a real field-of-use restriction. Apache-2.0 was
chosen as the base specifically to make everything *other* than that one
manual-review step as frictionless as possible.

## Self-use

The Licensor is exempt from the licence's own restrictions by construction —
a copyright holder does not need a licence from itself. No self-grant text
required; confirmed explicit in the LICENSE file's closing paragraph so a
future reader doesn't have to re-derive it.

## Checklist for landing this in the new repo

1. Copy `LICENSE` and `NOTICE` to the repo root.
2. Add an SPDX header to new/modified `.go` files if the project wants that
   convention: `// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-gollum-military-exclusion`
   (there is no registered SPDX id for the addendum — reference the LICENSE
   file by name instead if a formal identifier is wanted).
3. Update `README.md`'s installation section to mention the licence and link
   to `LICENSE` explicitly, since the addendum won't surface from a licence
   badge alone.
4. `go.mod` module path `github.com/networkinss/gollum` is already correct
   for this to become a real tagged module — no rename needed.
5. Decide the first tag only after `StreamingBackend` lands (see edith's
   `TODO.md` — the interface should be frozen before v0.x adopters show up).
6. Resolve the `go.mod`/`build-llm.sh` llama-go pin mismatch
   (`v0.0.0-20260409130703-...` vs. the vetted `b8a6878`) before publishing —
   a consumer building with `-tags llm` today would get the unvetted commit.
7. Drop the `replace github.com/tcpipuk/llama-go => ./third_party/llama-go`
   line from the published `go.mod`; keep local development wired through a
   `go.work` instead.
