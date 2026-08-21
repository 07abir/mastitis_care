"""Oracle consensus layer - the 2-of-3 agreement check.

A single model deciding a cow's fate is a weak foundation for a claim a buyer is
meant to trust. So before anything reaches the chain, three independent sources
vote, and at least two must agree:

  Source A - ML ensemble        the soft-voting model's prediction
  Source B - somatic cell count the international clinical benchmark for
                                subclinical mastitis, SCC above 200,000 cells/mL
  Source C - physical signs     clotting, alkaline pH, raised conductivity,
                                fever, and yield drop; two or more signs = positive

Why this shape: the three sources fail in different ways. The ML model can be
wrong on a cow unlike anything in training, SCC alone misses very early cases,
and physical signs alone are noisy. Requiring agreement means a single bad sensor
or an overconfident model cannot by itself put a false record on an immutable
ledger.

Crucially, "agreed" means the ML prediction is in the majority -- not just that
some majority exists. If both sensor-based sources contradict the model, the cow
is withheld for a vet rather than published either way. See `evaluate`.

The thresholds below come from standard dairy practice, not from fitting this
dataset -- which is what makes Source B and C genuinely independent of Source A
rather than three views of the same training data.
"""
from __future__ import annotations

from dataclasses import dataclass

# Somatic cell count, in thousands of cells/mL to match the dataset's units.
# The widely used subclinical-mastitis threshold is 200,000 cells/mL.
SCC_THRESHOLD = 200.0

# Healthy milk sits near pH 6.5-6.7; infection pushes it alkaline.
PH_THRESHOLD = 6.80
# Normal electrical conductivity is roughly 4.0-5.5 mS/cm; inflammation raises
# sodium and chloride, and with them conductivity.
CONDUCTIVITY_THRESHOLD = 5.50
# Local inflammation shows up as a warmer quarter.
TEMPERATURE_THRESHOLD = 37.50
# Clinical mastitis typically suppresses yield.
YIELD_DROP_THRESHOLD = 14.0

# How many physical signs constitute a positive Source C vote.
PHYSICAL_SIGNS_REQUIRED = 2


@dataclass
class ConsensusResult:
    agreed: bool          # is the ML prediction corroborated by the majority?
    verdict: int          # 0 healthy / 1 mastitic - the majority view
    votes: dict           # per-source vote
    votes_for: int        # how many sources voted mastitic
    reasons: list         # human-readable explanation for the dashboard
    unanimous: bool
    ml_outvoted: bool     # the sensors contradicted the model

    def as_dict(self):
        return {
            "consensus_agreed": self.agreed,
            "consensus_verdict": self.verdict,
            "consensus_votes": self.votes,
            "consensus_votes_for": self.votes_for,
            "consensus_unanimous": self.unanimous,
            "consensus_ml_outvoted": self.ml_outvoted,
            "consensus_reasons": self.reasons,
        }


def _get(row, *names):
    """Fetch the first present column, tolerating naming variations."""
    for n in names:
        if n in row and row[n] is not None:
            try:
                return float(row[n])
            except (TypeError, ValueError):
                continue
    return None


def source_b_scc(row):
    """Somatic cell count vote."""
    scc = _get(row, "Somatic_Cell_Count", "SCC", "scc")
    if scc is None:
        return None, "SCC unavailable"
    vote = int(scc >= SCC_THRESHOLD)
    return vote, f"SCC {scc:.0f}k cells/mL {'>=' if vote else '<'} {SCC_THRESHOLD:.0f}k threshold"


def source_c_physical(row):
    """Physical and chemical signs vote."""
    signs, notes = 0, []

    clot = _get(row, "Clotting", "clotting")
    if clot is not None and clot >= 1:
        signs += 1
        notes.append("clots present in milk")

    ph = _get(row, "Milk_pH", "pH", "ph")
    if ph is not None and ph > PH_THRESHOLD:
        signs += 1
        notes.append(f"pH {ph:.2f} above {PH_THRESHOLD}")

    ec = _get(row, "Milk_Conductivity", "EC", "conductivity")
    if ec is not None and ec > CONDUCTIVITY_THRESHOLD:
        signs += 1
        notes.append(f"conductivity {ec:.2f} above {CONDUCTIVITY_THRESHOLD} mS/cm")

    temp = _get(row, "Milk_Temperature", "temp", "temperature")
    if temp is not None and temp > TEMPERATURE_THRESHOLD:
        signs += 1
        notes.append(f"temperature {temp:.2f}C above {TEMPERATURE_THRESHOLD}C")

    yld = _get(row, "Milk_Yield", "yield")
    if yld is not None and yld < YIELD_DROP_THRESHOLD:
        signs += 1
        notes.append(f"yield {yld:.1f} L below {YIELD_DROP_THRESHOLD} L")

    vote = int(signs >= PHYSICAL_SIGNS_REQUIRED)
    if not notes:
        notes.append("no physical signs detected")
    return vote, f"{signs} physical sign(s): " + "; ".join(notes)


def evaluate(ml_prediction: int, row, ml_confidence=None) -> ConsensusResult:
    """Run the 2-of-3 check for one cow.

    ``row`` is the cow's raw (unscaled) reading as a dict or pandas Series.
    """
    votes, reasons = {}, []

    votes["ml_ensemble"] = int(ml_prediction)
    conf = f" (confidence {ml_confidence:.2%})" if ml_confidence is not None else ""
    reasons.append(
        f"ML ensemble: {'mastitic' if ml_prediction else 'healthy'}{conf}"
    )

    for key, fn in (("somatic_cell_count", source_b_scc), ("physical_signs", source_c_physical)):
        vote, why = fn(row)
        if vote is not None:
            votes[key] = vote
        reasons.append(f"{key.replace('_', ' ')}: {why}")

    cast = list(votes.values())
    votes_for = sum(cast)
    votes_against = len(cast) - votes_for

    verdict = int(votes_for > votes_against)
    majority = max(votes_for, votes_against)

    # `agreed` deliberately means "the ML prediction is in the majority", not
    # merely "a majority exists". If the two sensor-based sources outvote the
    # model, a majority technically exists but the model is the odd one out --
    # exactly the case this layer is here to catch. Publishing the model's call
    # anyway would defeat the purpose, and silently substituting the sensors'
    # verdict would put an unreviewed diagnosis on an immutable ledger. So the
    # cow is withheld for a vet instead.
    ml_vote = votes["ml_ensemble"]
    ml_outvoted = verdict != ml_vote
    agreed = (majority >= 2) and not ml_outvoted
    if len(cast) < 3:
        # A source was unavailable; with only two votes, require both to line up.
        agreed = (majority == len(cast)) and not ml_outvoted

    return ConsensusResult(
        agreed=agreed,
        verdict=verdict,
        votes=votes,
        votes_for=votes_for,
        reasons=reasons,
        unanimous=(votes_for == len(cast) or votes_against == len(cast)),
        ml_outvoted=ml_outvoted,
    )


if __name__ == "__main__":
    import pandas as pd

    from . import config

    df = pd.read_csv(config.DATA_CSV)
    print("2-of-3 consensus, using the CSV label as a stand-in for the ML vote\n")

    agree = disagree = 0
    for _, row in df.iterrows():
        r = evaluate(int(row[config.TARGET_COLUMN]), row)
        agree += r.agreed
        disagree += not r.agreed

    print(f"  ML corroborated    : {agree}/{len(df)} ({agree/len(df):.1%})")
    print(f"  held for vet review: {disagree}")

    print("\n  example cow from the dataset:")
    r = evaluate(int(df.iloc[0][config.TARGET_COLUMN]), df.iloc[0], 0.98)
    for line in r.reasons:
        print(f"    - {line}")
    print(f"    => verdict={r.verdict} agreed={r.agreed} unanimous={r.unanimous}")

    print("\n  synthetic cow where the model contradicts every sensor:")
    contradictory = {
        "Somatic_Cell_Count": 40, "Milk_pH": 6.50, "Milk_Conductivity": 4.20,
        "Milk_Temperature": 35.0, "Milk_Yield": 25.0, "Clotting": 0,
    }
    r = evaluate(1, contradictory, 0.99)  # model insists she is sick
    for line in r.reasons:
        print(f"    - {line}")
    print(f"    => verdict={r.verdict} agreed={r.agreed} ml_outvoted={r.ml_outvoted}")
    print("    (agreed=False means this cow goes to a vet, not to the chain)")
