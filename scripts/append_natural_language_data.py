#!/usr/bin/env python3
"""Script to append expanded natural-language and implicit commands to data/stt/commands.jsonl."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.intents.normalizer import normalize_text

NEW_RECORDS = [
    # Group 1: Global OFF
    {"id": "CMD_311", "utterance": "turn off everything", "category": "global", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Canonical global off"},
    {"id": "CMD_312", "utterance": "switch everything off", "category": "global", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Synonym verb switch everything off"},
    {"id": "CMD_313", "utterance": "shut everything down", "category": "global", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Synonym verb shut down everything"},
    {"id": "CMD_314", "utterance": "switch it all off", "category": "global", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Collective pronoun it all off"},
    {"id": "CMD_315", "utterance": "turn all the systems off", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Collective noun phrase all the systems off"},
    {"id": "CMD_316", "utterance": "turn off all the vehicle systems", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Formal vehicle systems off"},
    {"id": "CMD_317", "utterance": "deactivate everything", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Formal deactivate everything"},
    {"id": "CMD_318", "utterance": "switch off all the electronics", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Collective electronics off"},
    {"id": "CMD_319", "utterance": "power down all controllable systems", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Power down controllable systems"},
    {"id": "CMD_320", "utterance": "i want everything switched off", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Conversational desire everything off"},
    {"id": "CMD_321", "utterance": "turn off the electricals", "category": "global", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Colloquial electricals off"},
    {"id": "CMD_322", "utterance": "shut down the cabin systems", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Cabin systems shutdown"},
    {"id": "CMD_323", "utterance": "turn off the nonessential stuff", "category": "global", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Colloquial nonessential off"},
    {"id": "CMD_324", "utterance": "turn off all the lights and climate controls", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Compound collective lights and climate"},
    {"id": "CMD_325", "utterance": "everything off, please", "category": "global", "split": "dev", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Polite telegraphic everything off"},
    {"id": "CMD_326", "utterance": "can you shut everything down", "category": "global", "split": "dev", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Polite question everything down"},
    {"id": "CMD_327", "utterance": "turn off all controllable systems", "category": "global", "split": "dev", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Controllable systems off"},
    {"id": "CMD_328", "utterance": "disable all vehicle systems", "category": "global", "split": "dev", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Disable all systems"},
    {"id": "CMD_329", "utterance": "everything off", "category": "global", "split": "test", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Telegraphic held-out everything off"},
    {"id": "CMD_330", "utterance": "please switch off everything", "category": "global", "split": "test", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Polite prefix held-out switch off everything"},
    {"id": "CMD_331", "utterance": "turn off all systems", "category": "global", "split": "test", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "All systems off held-out"},
    {"id": "CMD_332", "utterance": "shut down everything", "category": "global", "split": "test", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_OFF", "parameters": {}}], "notes": "Shut down everything held-out"},

    # Group 2: Global ON
    {"id": "CMD_333", "utterance": "turn everything on", "category": "global", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Canonical global on"},
    {"id": "CMD_334", "utterance": "switch everything on", "category": "global", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Synonym verb switch everything on"},
    {"id": "CMD_335", "utterance": "enable everything", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Formal enable everything"},
    {"id": "CMD_336", "utterance": "turn everything back on", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Restoration everything back on"},
    {"id": "CMD_337", "utterance": "restore all systems", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Restore all systems"},
    {"id": "CMD_338", "utterance": "enable all the systems again", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Enable all systems again"},
    {"id": "CMD_339", "utterance": "activate all vehicle systems", "category": "global", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Activate vehicle systems"},
    {"id": "CMD_340", "utterance": "turn on all controllable systems", "category": "global", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "All controllable systems on"},
    {"id": "CMD_341", "utterance": "switch all systems on", "category": "global", "split": "dev", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Switch all systems on"},
    {"id": "CMD_342", "utterance": "power on all systems", "category": "global", "split": "dev", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Power on all systems"},
    {"id": "CMD_343", "utterance": "turn on everything", "category": "global", "split": "test", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Held-out turn on everything"},
    {"id": "CMD_344", "utterance": "everything on", "category": "global", "split": "test", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "GLOBAL_ON", "parameters": {}}], "notes": "Held-out telegraphic everything on"},

    # Group 3: Contextual & Clarification Queries
    {"id": "CMD_345", "utterance": "it is too dark in here", "category": "assistant", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "headlights"}}], "notes": "Clarify headlights suggestion"},
    {"id": "CMD_346", "utterance": "i cannot see anything", "category": "assistant", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "headlights"}}], "notes": "Clarify headlights inquiry"},
    {"id": "CMD_347", "utterance": "it is getting hot inside", "category": "assistant", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "ac"}}], "notes": "Clarify AC suggestion"},
    {"id": "CMD_348", "utterance": "the car is getting warm", "category": "assistant", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "ac"}}], "notes": "Clarify AC temperature prompt"},
    {"id": "CMD_349", "utterance": "the cabin is too humid", "category": "assistant", "split": "train", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "humidity"}}], "notes": "Clarify humidity sensor query"},
    {"id": "CMD_350", "utterance": "it feels too humid in here", "category": "assistant", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "humidity"}}], "notes": "Clarify humidity sensory report"},
    {"id": "CMD_351", "utterance": "make the cabin more comfortable", "category": "assistant", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "comfort"}}], "notes": "Clarify comfort multi-option prompt"},
    {"id": "CMD_352", "utterance": "it is dark outside", "category": "assistant", "split": "dev", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "headlights"}}], "notes": "Clarify headlights outside dark"},
    {"id": "CMD_353", "utterance": "the cabin feels hot", "category": "assistant", "split": "dev", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "ac"}}], "notes": "Clarify AC cabin hot"},
    {"id": "CMD_354", "utterance": "is it too humid in here", "category": "assistant", "split": "dev", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "humidity"}}], "notes": "Clarify humidity inquiry question"},
    {"id": "CMD_355", "utterance": "it's too dark in here", "category": "assistant", "split": "test", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "headlights"}}], "notes": "Held-out dark in here with contraction"},
    {"id": "CMD_356", "utterance": "it's getting hot inside", "category": "assistant", "split": "test", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "ac"}}], "notes": "Held-out hot inside with contraction"},
    {"id": "CMD_357", "utterance": "the car is too warm", "category": "assistant", "split": "test", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "ac"}}], "notes": "Held-out car too warm"},
    {"id": "CMD_358", "utterance": "make the cabin comfortable", "category": "assistant", "split": "test", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "CLARIFY", "parameters": {"target": "comfort"}}], "notes": "Held-out comfort clarification"},

    # Group 4: Compound Commands with Negation, Exclusions, Anaphora
    {"id": "CMD_359", "utterance": "turn off everything except the headlights", "category": "compound", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "EMERGENCY_OFF", "parameters": {}}], "notes": "Global exclusion of headlights"},
    {"id": "CMD_360", "utterance": "leave the lights on and switch everything else off", "category": "compound", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "EMERGENCY_OFF", "parameters": {}}], "notes": "Lights preserved on; everything else off"},
    {"id": "CMD_361", "utterance": "turn off the ac, not the headlights", "category": "compound", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "TURN_AC_OFF", "parameters": {}}], "notes": "Contrastive negation headlights excluded"},
    {"id": "CMD_362", "utterance": "turn the lights and ac off", "category": "compound", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "notes": "Collective lights and AC off"},
    {"id": "CMD_363", "utterance": "turn on the lights and leave them on", "category": "compound", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "LIGHTS_ON", "parameters": {}}], "notes": "State preservation leave them on"},
    {"id": "CMD_364", "utterance": "switch on the ac and turn it off again", "category": "compound", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "notes": "Sequential AC on then off"},
    {"id": "CMD_365", "utterance": "open the driver door and then close it", "category": "compound", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}, {"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "notes": "Pronoun it refers to driver door"},
    {"id": "CMD_366", "utterance": "turn the ac on and leave everything else alone", "category": "compound", "split": "train", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "TURN_AC_ON", "parameters": {}}], "notes": "AC isolated from other systems"},
    {"id": "CMD_367", "utterance": "don't touch the doors; turn the lights off", "category": "compound", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "LIGHTS_OFF", "parameters": {}}], "notes": "Doors excluded; lights off"},
    {"id": "CMD_368", "utterance": "close the driver door but leave the passenger door open", "category": "compound", "split": "train", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "notes": "Driver door closed; passenger door open"},
    {"id": "CMD_369", "utterance": "turn off everything except the ac", "category": "compound", "split": "dev", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "EMERGENCY_OFF", "parameters": {}}], "notes": "Dev global exclusion of AC"},
    {"id": "CMD_370", "utterance": "turn on the ac, not the headlights", "category": "compound", "split": "dev", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "TURN_AC_ON", "parameters": {}}], "notes": "Dev contrastive AC on"},
    {"id": "CMD_371", "utterance": "lights and ac off", "category": "compound", "split": "dev", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "notes": "Dev telegraphic lights and AC off"},
    {"id": "CMD_372", "utterance": "turn off the ac but not the lights", "category": "compound", "split": "test", "difficulty": "medium", "supported": True, "expected_actions": [{"type": "TURN_AC_OFF", "parameters": {}}], "notes": "Held-out contrastive AC off lights excluded"},
    {"id": "CMD_373", "utterance": "turn on the headlights and ac", "category": "compound", "split": "test", "difficulty": "easy", "supported": True, "expected_actions": [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_ON", "parameters": {}}], "notes": "Held-out headlights and AC on"},
    {"id": "CMD_374", "utterance": "turn on the ac and then turn it off", "category": "compound", "split": "test", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "notes": "Held-out sequential AC cycle"},
    {"id": "CMD_375", "utterance": "open the driver door and close it", "category": "compound", "split": "test", "difficulty": "hard", "supported": True, "expected_actions": [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}, {"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "notes": "Held-out pronoun door cycle"},

    # Group 5: Unsupported, Contradictions, and Context-Missing Ambiguities
    {"id": "CMD_376", "utterance": "turn off the ac but keep it running", "category": "unsupported", "split": "train", "difficulty": "hard", "supported": False, "expected_actions": [], "notes": "Contradiction rejection AC off vs running"},
    {"id": "CMD_377", "utterance": "turn on the lights and leave them off", "category": "unsupported", "split": "train", "difficulty": "hard", "supported": False, "expected_actions": [], "notes": "Contradiction rejection lights on vs off"},
    {"id": "CMD_378", "utterance": "switch both of those off", "category": "unsupported", "split": "train", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Ambiguity rejection without prior context"},
    {"id": "CMD_379", "utterance": "both on", "category": "unsupported", "split": "train", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Ambiguity rejection without prior context"},
    {"id": "CMD_380", "utterance": "now the other one", "category": "unsupported", "split": "train", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Ambiguity rejection without prior context"},
    {"id": "CMD_381", "utterance": "the headlights too", "category": "unsupported", "split": "train", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Ambiguity rejection without prior context"},
    {"id": "CMD_382", "utterance": "set cabin temperature to eighteen degrees", "category": "unsupported", "split": "train", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Unsupported temperature setting"},
    {"id": "CMD_383", "utterance": "dehumidify the vehicle", "category": "unsupported", "split": "train", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Unsupported dehumidifier actuator"},
    {"id": "CMD_384", "utterance": "turn on the headlights but keep them off", "category": "unsupported", "split": "dev", "difficulty": "hard", "supported": False, "expected_actions": [], "notes": "Contradiction rejection lights on vs off"},
    {"id": "CMD_385", "utterance": "switch the two systems back on", "category": "unsupported", "split": "dev", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Ambiguity rejection without prior context"},
    {"id": "CMD_386", "utterance": "turn the ac off but keep it running", "category": "unsupported", "split": "test", "difficulty": "hard", "supported": False, "expected_actions": [], "notes": "Held-out contradiction rejection AC off vs running"},
    {"id": "CMD_387", "utterance": "disable both systems", "category": "unsupported", "split": "test", "difficulty": "medium", "supported": False, "expected_actions": [], "notes": "Held-out ambiguity rejection without antecedent"},
]

def main():
    dataset_path = REPO_ROOT / "data" / "stt" / "commands.jsonl"
    
    # Read existing records
    existing_ids = set()
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                existing_ids.add(rec["id"])

    to_append = [r for r in NEW_RECORDS if r["id"] not in existing_ids]

    for r in to_append:
        r["normalized_utterance"] = normalize_text(r["utterance"])

    with open(dataset_path, "a", encoding="utf-8") as f:
        for r in to_append:
            f.write(json.dumps(r) + "\n")

    print(f"Appended {len(to_append)} new records to {dataset_path}.")

if __name__ == "__main__":
    main()
