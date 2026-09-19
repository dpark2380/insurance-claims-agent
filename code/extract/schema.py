"""Structured extraction schema for AU home-insurance claim narratives.

Single source of truth for both the zero-shot prompt and the LoRA fine-tune's
target format -- both must produce JSON that validates against this model.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    # STORM vs FLOOD follows the AU standard flood definition (Insurance Contracts
    # Act, standard cover terms since 2012): FLOOD is water escaping the normal
    # confines of a lake, river, creek, dam, or other natural watercourse. Any other
    # rain-caused loss -- wind-driven rain, roof ingress, drainage overwhelm, surface
    # pooling not from a named watercourse -- is STORM, regardless of rainfall volume.
    STORM = "storm"
    FLOOD = "flood"
    FIRE = "fire"
    THEFT = "theft"
    ACCIDENTAL_DAMAGE = "accidental_damage"
    ESCAPE_OF_LIQUID = "escape_of_liquid"
    MALICIOUS_DAMAGE = "malicious_damage"
    GLASS_BREAKAGE = "glass_breakage"
    OTHER = "other"


class ClaimExtraction(BaseModel):
    claim_type: ClaimType
    date_of_loss: date
    cause: str = Field(..., description="Short free-text cause, e.g. 'burst pipe under kitchen sink'")
    damaged_item: str = Field(..., description="What was damaged, e.g. 'kitchen ceiling and cabinetry'")
    estimated_amount: float | None = Field(None, description="Claimant's estimated repair/replacement cost in AUD")
    policy_number: str | None = Field(None, description="Policy number if mentioned in the narrative, else null")


if __name__ == "__main__":
    # Smoke test: 3 hand-crafted narrative -> expected-extraction pairs, so schema
    # design bugs surface before generating 150 synthetic examples against it.
    examples = [
        (
            "A storm on 14 March 2025 ripped several tiles off our roof, and rain "
            "got into the ceiling cavity, damaging the plasterboard in the main "
            "bedroom. We estimate repairs at around $6,200. Policy number "
            "AAM-88213-H.",
            {
                "claim_type": "storm",
                "date_of_loss": "2025-03-14",
                "cause": "storm ripped tiles off roof, rain entered ceiling cavity",
                "damaged_item": "roof tiles and bedroom ceiling plasterboard",
                "estimated_amount": 6200.0,
                "policy_number": "AAM-88213-H",
            },
        ),
        (
            "Someone broke into our garage last night (2025-06-02) and took a "
            "mountain bike and a set of power tools. We haven't had a chance to "
            "get a full quote yet.",
            {
                "claim_type": "theft",
                "date_of_loss": "2025-06-02",
                "cause": "break-in to garage",
                "damaged_item": "mountain bike and power tools",
                "estimated_amount": None,
                "policy_number": None,
            },
        ),
        (
            "On 2025-01-20 our washing machine's inlet hose burst overnight and "
            "flooded the laundry and the hallway carpet next to it. Repair quote "
            "came in at $3,450.50.",
            {
                "claim_type": "escape_of_liquid",
                "date_of_loss": "2025-01-20",
                "cause": "washing machine inlet hose burst",
                "damaged_item": "laundry flooring and hallway carpet",
                "estimated_amount": 3450.50,
                "policy_number": None,
            },
        ),
    ]

    for narrative, expected in examples:
        parsed = ClaimExtraction.model_validate(expected)
        print(f"OK  {parsed.claim_type.value:20s} {parsed.date_of_loss}  ${parsed.estimated_amount}")
        assert parsed.model_dump(mode="json")["claim_type"] == expected["claim_type"]
    print(f"\n{len(examples)}/{len(examples)} smoke-test examples validated against ClaimExtraction")
