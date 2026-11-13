"""Where a result or figure file lives, now that each idea owns its folder.

The reorganisation of 2026-11-13 split one `results/` directory into four --
one per idea, plus `core/results/` for project-wide artefacts -- and one
`figures/` directory into three. Scripts that span more than one idea cannot
name a single directory any more, so they ask here instead:

    core/analysis/final.py     assembles RESULTS.md from all three ideas
    core/analysis/figures.py   draws Idea A and Idea B side by side
    conditioning/analysis/figures_cond.py   reuses that palette, own figures

Ownership follows the result-file prefix, which is how this project has
labelled its findings from the start:

    A, Ac, Am, C, D, E, F, H    Idea A     contact-gated parallel unmasking
    B                           Idea B     the two-level chemical alphabet
    G, I, J, K, L, M            conditioning

Three files carry a prefix that predates the experiment they belong to, and
are listed in OVERRIDE: the first conditioning run was numbered under B_, and
the withdrawn beta-fold note under G_.

Lookup is forgiving on purpose. `result_path` returns the first directory
that actually holds the file and falls back to the prefix rule, so callers
keep their "skip this panel if the input is missing" behaviour unchanged.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IDEA_A = "idea_a_contacts"
IDEA_B = "idea_b_two_level"
COND = "conditioning"
CORE = "core"

OWNER_BY_PREFIX = {
    "A": IDEA_A, "Ac": IDEA_A, "Am": IDEA_A, "C": IDEA_A,
    "D": IDEA_A, "E": IDEA_A, "F": IDEA_A, "H": IDEA_A,
    "B": IDEA_B,
    "G": COND, "I": COND, "J": COND, "K": COND, "L": COND, "M": COND,
}
OVERRIDE = {
    "B_cond.json": COND,            # the first conditioning run, numbered under B_
    "B_cond_scored.json": COND,
    "B_conditioning_note.md": COND,
    "G_beta_effect_note.md": IDEA_A,  # the withdrawn Idea-A fold-class finding
    "smoke.json": CORE,
    "EXTERNAL_BASELINES.md": CORE,
}
FIGURE_OWNER = {
    "fig1_contact_quality": IDEA_A,
    "fig2_plddt_vs_nfe": IDEA_A,
    "fig3_contrasts": IDEA_A,
    "fig5_oracle": IDEA_A,
    "fig4_ideaB": IDEA_B,
    "fig6_ideaB": IDEA_B,
    "fig7_conditioning": COND,
    "fig8_pattern_budget": COND,
    "fig9_independent_metrics": COND,
}
RESULT_DIRS = tuple(os.path.join(ROOT, d, "results")
                    for d in (IDEA_A, IDEA_B, COND, CORE))


def result_owner(name):
    """Which idea folder owns this result file, by name alone."""
    if name in OVERRIDE:
        return OVERRIDE[name]
    return OWNER_BY_PREFIX.get(name.split("_")[0], CORE)


def result_dir(owner):
    return os.path.join(ROOT, owner, "results")


def result_path(name):
    """Full path to a result file: where it is, else where it belongs."""
    for d in RESULT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(result_dir(result_owner(name)), name)


def figure_dir(stem):
    """Directory for a figure, created if it does not exist yet."""
    owner = FIGURE_OWNER.get(stem, CORE)
    d = os.path.join(ROOT, owner, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def figure_path(stem, ext):
    """Full path for one figure file, e.g. figure_path('fig5_oracle', 'pdf')."""
    return os.path.join(figure_dir(stem), f"{stem}.{ext}")
