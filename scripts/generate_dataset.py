#!/usr/bin/env python3
"""Dataset generator for vehicle voice assistant STT and Intent evaluation.

Generates:
- data/stt/commands.jsonl (canonical structured JSONL)
- data/stt/commands.csv (human-readable CSV export)
"""

import csv
import json
import os
from typing import List, Dict, Any

from app.intents.normalizer import normalize_text


def create_dataset_records() -> List[Dict[str, Any]]:
    """Constructs the canonical 300-utterance evaluation dataset."""
    raw_data = [
        # =====================================================================
        # A. Headlights and Lighting (36 items)
        # =====================================================================
        # Train
        ("turn on the headlights", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "easy", True, "Canonical lights on"),
        ("switch on the headlights", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "easy", True, "Synonym verb switch on"),
        ("activate the headlights", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "easy", True, "Formal verb activate"),
        ("turn the lights on, please", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "easy", True, "Split verb order with politeness"),
        ("headlights on", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "easy", True, "Telegraphic command"),
        ("turn on the car lights", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "easy", True, "Alternative noun phrase"),
        ("power on the headlights", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "easy", True, "Synonym power on"),
        ("hey assistant, please turn on the headlights", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "medium", True, "Wake phrase prefix with politeness"),
        ("turn off the headlights", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "easy", True, "Canonical lights off"),
        ("switch off the lights", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "easy", True, "Synonym switch off"),
        ("deactivate the headlights", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "easy", True, "Formal deactivate"),
        ("lights off", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "easy", True, "Telegraphic lights off"),
        ("shut off the headlights", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "easy", True, "Synonym shut off"),
        ("could you turn off the vehicle lights", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "medium", True, "Polite request question form"),
        ("are the headlights on", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "train", "easy", True, "Direct state question"),
        ("what is the current light status", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "train", "easy", True, "Explicit status query"),
        ("check the headlights", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "train", "easy", True, "Imperative check query"),
        ("keep the headlights on", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "medium", True, "State preservation verb keep"),
        ("leave the lights off", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "medium", True, "State preservation verb leave"),
        ("enable vehicle lighting", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "train", "medium", True, "Alternative formal lighting noun"),
        ("disable vehicle lighting", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "train", "medium", True, "Alternative formal lighting off"),
        ("are the vehicle lights switched off", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "train", "medium", True, "Interrogative passive check"),

        # Dev
        ("switch the headlights on right now", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "dev", "medium", True, "Split verb with urgency suffix"),
        ("turn headlights off immediately", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "dev", "medium", True, "Immediate action suffix"),
        ("is the lighting active", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "dev", "medium", True, "Formal inquiry"),
        ("please leave the headlights on", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "dev", "medium", True, "Polite state preservation"),
        ("keep the lamps off", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "dev", "medium", True, "Lamps synonym with keep"),
        ("check headlight status", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "dev", "easy", True, "Terse query"),
        ("can you switch on the lamps", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "dev", "medium", True, "Polite lamps request"),

        # Test (Held-out)
        ("turn the headlights on for me", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "test", "medium", True, "Conversational beneficiary suffix"),
        ("shut the lights down", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "test", "medium", True, "Phrasal verb shut down"),
        ("are my headlights currently turned on", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "test", "medium", True, "Possessive pronoun question"),
        ("headlight status please", [{"type": "GET_LIGHTS_STATUS", "parameters": {}}], "lighting", "test", "easy", True, "Terse query with politeness"),
        ("leave the headlights running", [{"type": "LIGHTS_ON", "parameters": {}}], "lighting", "test", "medium", True, "Preservation variation"),
        ("make sure the lights stay off", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "test", "hard", True, "Stay off variation"),
        ("switch off the exterior lights", [{"type": "LIGHTS_OFF", "parameters": {}}], "lighting", "test", "medium", True, "Exterior modifier"),

        # =====================================================================
        # B. Air Conditioning (36 items)
        # =====================================================================
        # Train
        ("turn on the ac", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "easy", True, "Canonical AC on"),
        ("switch on the air conditioning", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "easy", True, "Full noun air conditioning"),
        ("start the air conditioner", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "easy", True, "Verb start with air conditioner"),
        ("activate the ac", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "easy", True, "Formal activate"),
        ("turn off the ac", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "easy", True, "Canonical AC off"),
        ("switch off the air conditioning", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "easy", True, "Full noun air conditioning off"),
        ("stop the air conditioner", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "easy", True, "Verb stop air conditioner"),
        ("is the ac running", [{"type": "GET_AC_STATUS", "parameters": {}}], "climate", "train", "easy", True, "Interrogative running check"),
        ("check the ac status", [{"type": "GET_AC_STATUS", "parameters": {}}], "climate", "train", "easy", True, "Explicit status check"),
        ("keep the ac running", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "medium", True, "Keep AC running preservation"),
        ("leave the ac off", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "medium", True, "Leave AC off preservation"),
        ("it is too warm in here turn on the ac", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "medium", True, "Causal preface warm cabin"),
        ("it is too hot turn on cooling", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "medium", True, "Causal preface hot with cooling"),
        ("turn the ac off, please", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "easy", True, "Split verb order with politeness"),
        ("power on the air conditioning", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "easy", True, "Power on climate"),
        ("shut off the climate control", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "medium", True, "Climate control shut off"),
        ("enable cabin cooling", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "medium", True, "Enable cabin cooling"),
        ("disable the ac", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "easy", True, "Disable verb"),
        ("ac status", [{"type": "GET_AC_STATUS", "parameters": {}}], "climate", "train", "easy", True, "Terse ac status"),
        ("ac on", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "train", "easy", True, "Terse ac on"),
        ("ac off", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "train", "easy", True, "Terse ac off"),

        # Dev
        ("start cooling the cabin", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "dev", "medium", True, "Gerund cooling phrase"),
        ("stop the cooling system", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "dev", "medium", True, "Cooling system noun"),
        ("what is the ac status", [{"type": "GET_AC_STATUS", "parameters": {}}], "climate", "dev", "easy", True, "What is status inquiry"),
        ("is the climate control active", [{"type": "GET_AC_STATUS", "parameters": {}}], "climate", "dev", "medium", True, "Climate control query"),
        ("can you please turn on the air conditioner", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "dev", "medium", True, "Polite request modal"),
        ("turn off the air conditioning for me", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "dev", "medium", True, "Beneficiary suffix"),
        ("keep cooling active", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "dev", "medium", True, "Cooling active state preservation"),

        # Test (Held-out)
        ("turn the ac on right now", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "test", "medium", True, "Urgency adverbial"),
        ("shut down the air conditioning", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "test", "medium", True, "Shut down phrasal verb"),
        ("is the air conditioning currently turned on", [{"type": "GET_AC_STATUS", "parameters": {}}], "climate", "test", "medium", True, "Adverbial query currently"),
        ("it is getting warm please start the ac", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "test", "medium", True, "Progressive temperature preface"),
        ("leave the air conditioning switched off", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "test", "medium", True, "Switched off passive preservation"),
        ("check if the ac is running", [{"type": "GET_AC_STATUS", "parameters": {}}], "climate", "test", "medium", True, "Embedded conditional clause"),
        ("power off the cooling unit", [{"type": "TURN_AC_OFF", "parameters": {}}], "climate", "test", "medium", True, "Cooling unit phrasing"),
        ("activate cabin air conditioning please", [{"type": "TURN_AC_ON", "parameters": {}}], "climate", "test", "medium", True, "Cabin modifier with politeness"),

        # =====================================================================
        # C. Door Controls (42 items)
        # =====================================================================
        # Train
        ("open the driver door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "doors", "train", "easy", True, "Open driver door"),
        ("open the driver's side door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "doors", "train", "easy", True, "Possessive driver's side"),
        ("open the left door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "doors", "train", "easy", True, "Left directional mapping"),
        ("close the driver door", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "doors", "train", "easy", True, "Close driver door"),
        ("shut the driver's door", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "doors", "train", "easy", True, "Verb shut driver door"),
        ("close the left door", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "doors", "train", "easy", True, "Left directional close"),
        ("open the passenger door", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "doors", "train", "easy", True, "Open passenger door"),
        ("open the passenger's side door", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "doors", "train", "easy", True, "Possessive passenger's side"),
        ("open the right door", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "doors", "train", "easy", True, "Right directional mapping"),
        ("close the passenger door", [{"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "doors", "train", "easy", True, "Close passenger door"),
        ("shut the passenger's door", [{"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "doors", "train", "easy", True, "Verb shut passenger door"),
        ("close the right door", [{"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "doors", "train", "easy", True, "Right directional close"),
        ("open both doors", [{"type": "OPEN_DOOR", "parameters": {"door": "both"}}], "doors", "train", "easy", True, "Both doors open"),
        ("close both doors", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "doors", "train", "easy", True, "Both doors close"),
        ("shut both doors", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "doors", "train", "easy", True, "Shut both doors"),
        ("open all doors", [{"type": "OPEN_DOOR", "parameters": {"door": "both"}}], "doors", "train", "easy", True, "All doors open"),
        ("close all doors", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "doors", "train", "easy", True, "All doors close"),
        ("is the driver door open", [{"type": "GET_DOOR_STATUS", "parameters": {"door": "driver"}}], "doors", "train", "easy", True, "Driver door status query"),
        ("is the passenger door open", [{"type": "GET_DOOR_STATUS", "parameters": {"door": "passenger"}}], "doors", "train", "easy", True, "Passenger door status query"),
        ("check door status", [{"type": "GET_DOOR_STATUS", "parameters": {}}], "doors", "train", "easy", True, "General door status check"),
        ("check the doors", [{"type": "GET_DOOR_STATUS", "parameters": {}}], "doors", "train", "easy", True, "Check doors query"),
        ("door status", [{"type": "GET_DOOR_STATUS", "parameters": {}}], "doors", "train", "easy", True, "Terse door status"),
        # Ambiguous single door commands
        ("open the door", [{"type": "OPEN_DOOR", "parameters": {"door": "unspecified"}}], "doors", "train", "medium", True, "Single door unspecified default behavior"),
        ("close the door", [{"type": "CLOSE_DOOR", "parameters": {"door": "unspecified"}}], "doors", "train", "medium", True, "Single door close unspecified default"),

        # Dev
        ("could you open the driver door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "doors", "dev", "medium", True, "Polite modal request"),
        ("please shut the passenger door", [{"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "doors", "dev", "medium", True, "Polite close request"),
        ("open driver side door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "doors", "dev", "easy", True, "Driver side without apostrophe"),
        ("close passenger side door", [{"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "doors", "dev", "easy", True, "Passenger side without apostrophe"),
        ("are both doors closed", [{"type": "GET_DOOR_STATUS", "parameters": {"door": "both"}}], "doors", "dev", "easy", True, "Both doors closed query"),
        ("are the vehicle doors open", [{"type": "GET_DOOR_STATUS", "parameters": {}}], "doors", "dev", "medium", True, "Vehicle doors query"),
        ("open the car door please", [{"type": "OPEN_DOOR", "parameters": {"door": "unspecified"}}], "doors", "dev", "medium", True, "Ambiguous car door request"),
        ("shut the car door", [{"type": "CLOSE_DOOR", "parameters": {"door": "unspecified"}}], "doors", "dev", "medium", True, "Ambiguous shut car door"),

        # Test (Held-out)
        ("open the left side door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "doors", "test", "medium", True, "Left side phrase"),
        ("shut the left side door", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "doors", "test", "medium", True, "Shut left side phrase"),
        ("open the right side door", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "doors", "test", "medium", True, "Right side phrase"),
        ("shut the right side door", [{"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "doors", "test", "medium", True, "Shut right side phrase"),
        ("can you open both vehicle doors", [{"type": "OPEN_DOOR", "parameters": {"door": "both"}}], "doors", "test", "medium", True, "Polite both doors request"),
        ("close both vehicle doors please", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "doors", "test", "medium", True, "Polite both doors close"),
        ("check if the driver door is open", [{"type": "GET_DOOR_STATUS", "parameters": {"door": "driver"}}], "doors", "test", "medium", True, "Check if conditional query"),
        ("is the passenger door shut", [{"type": "GET_DOOR_STATUS", "parameters": {"door": "passenger"}}], "doors", "test", "medium", True, "Shut state inquiry"),
        ("open the door, please", [{"type": "OPEN_DOOR", "parameters": {"door": "unspecified"}}], "doors", "test", "medium", True, "Ambiguous open door request with comma"),
        ("shut the door please", [{"type": "CLOSE_DOOR", "parameters": {"door": "unspecified"}}], "doors", "test", "medium", True, "Ambiguous shut door request"),

        # =====================================================================
        # D. Sensor Queries (36 items)
        # =====================================================================
        # Train
        ("what is the cabin temperature", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "train", "easy", True, "Canonical cabin temperature"),
        ("tell me the temperature", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "train", "easy", True, "Imperative tell me temperature"),
        ("check the temperature", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "train", "easy", True, "Check temperature"),
        ("how hot is it inside", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "train", "easy", True, "How hot inside inquiry"),
        ("how warm is the cabin", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "train", "easy", True, "How warm inquiry"),
        ("current temperature", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "train", "easy", True, "Terse temperature"),
        ("what is the temp in the car", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "train", "medium", True, "Abbreviated temp"),
        ("check the humidity", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "train", "easy", True, "Canonical humidity"),
        ("what is the humidity", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "train", "easy", True, "What is humidity query"),
        ("how humid is it inside", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "train", "easy", True, "How humid inquiry"),
        ("moisture level", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "train", "easy", True, "Moisture level phrasing"),
        ("cabin humidity", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "train", "easy", True, "Terse cabin humidity"),
        ("how far away is the obstacle", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "train", "easy", True, "Canonical distance query"),
        ("check the distance sensor", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "train", "easy", True, "Distance sensor check"),
        ("how close is the wall", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "train", "easy", True, "Wall obstacle inquiry"),
        ("check the distance", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "train", "easy", True, "Check distance"),
        ("proximity reading", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "train", "easy", True, "Proximity reading"),
        ("check all the sensors", [{"type": "GET_SENSOR_STATUS", "parameters": {}}], "sensors", "train", "easy", True, "All sensors check"),
        ("give me the sensor status", [{"type": "GET_SENSOR_STATUS", "parameters": {}}], "sensors", "train", "easy", True, "Sensor status request"),
        ("report the current sensor readings", [{"type": "GET_SENSOR_STATUS", "parameters": {}}], "sensors", "train", "easy", True, "Report readings request"),
        ("are the sensors working", [{"type": "GET_SENSOR_STATUS", "parameters": {}}], "sensors", "train", "easy", True, "Sensors health check"),
        ("sensor diagnostics", [{"type": "GET_SENSOR_STATUS", "parameters": {}}], "sensors", "train", "easy", True, "Sensor diagnostics"),

        # Dev
        ("what is the indoor temperature", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "dev", "medium", True, "Indoor modifier"),
        ("how cold is the car", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "dev", "easy", True, "How cold inquiry"),
        ("check moisture level in the cabin", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "dev", "medium", True, "Moisture level in cabin"),
        ("how humid is the cabin", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "dev", "easy", True, "Cabin humidity inquiry"),
        ("range to obstacle", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "dev", "easy", True, "Range phrasing"),
        ("proximity distance check", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "dev", "medium", True, "Proximity distance check"),
        ("scan sensors", [{"type": "GET_SENSOR_STATUS", "parameters": {}}], "sensors", "dev", "easy", True, "Scan sensors query"),

        # Test (Held-out)
        ("tell me the current cabin temperature", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "test", "medium", True, "Tell me current cabin temp"),
        ("how hot is the car right now", [{"type": "GET_TEMPERATURE", "parameters": {}}], "sensors", "test", "medium", True, "How hot right now"),
        ("what is the current humidity percentage", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "test", "medium", True, "Humidity percentage query"),
        ("check the moisture sensor please", [{"type": "GET_HUMIDITY", "parameters": {}}], "sensors", "test", "medium", True, "Moisture sensor check"),
        ("how far is the nearest obstacle", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "test", "medium", True, "Nearest obstacle phrasing"),
        ("distance to front obstacle", [{"type": "GET_DISTANCE", "parameters": {}}], "sensors", "test", "medium", True, "Front obstacle modifier"),
        ("give me full sensor diagnostics", [{"type": "GET_SENSOR_STATUS", "parameters": {}}], "sensors", "test", "medium", True, "Full sensor diagnostics"),

        # =====================================================================
        # E. Emergency Hazard Mode (18 items)
        # =====================================================================
        # Train
        ("emergency mode", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "train", "easy", True, "Terse emergency"),
        ("activate emergency mode", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "train", "easy", True, "Activate emergency"),
        ("turn on emergency mode", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "train", "easy", True, "Turn on emergency"),
        ("enable hazard lights", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "train", "easy", True, "Enable hazard lights"),
        ("hazard on", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "train", "easy", True, "Terse hazard on"),
        ("emergency off", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "train", "easy", True, "Terse emergency off"),
        ("disable emergency mode", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "train", "easy", True, "Disable emergency"),
        ("turn off hazard lights", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "train", "easy", True, "Turn off hazard lights"),
        ("switch off emergency system", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "train", "medium", True, "Switch off emergency system"),
        ("hazard off", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "train", "easy", True, "Terse hazard off"),
        ("stop hazard lights", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "train", "easy", True, "Stop hazard lights"),

        # Dev
        ("start hazard mode", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "dev", "easy", True, "Start hazard mode"),
        ("turn on the hazard warning lights", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "dev", "medium", True, "Warning lights phrasing"),
        ("deactivate hazard mode", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "dev", "medium", True, "Deactivate hazard"),
        ("turn off the emergency flasher", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "dev", "medium", True, "Flasher phrasing"),

        # Test (Held-out)
        ("switch on emergency hazard mode", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "test", "medium", True, "Switch on hazard mode"),
        ("please enable the hazard lights immediately", [{"type": "EMERGENCY_ON", "parameters": {}}], "emergency", "test", "medium", True, "Polite immediate hazard on"),
        ("disable the hazard warning system", [{"type": "EMERGENCY_OFF", "parameters": {}}], "emergency", "test", "medium", True, "Warning system disable"),

        # =====================================================================
        # F. General Assistant & Session Control (30 items)
        # =====================================================================
        # Train
        ("hello", [{"type": "GREETING", "parameters": {}}], "assistant", "train", "easy", True, "Greeting hello"),
        ("hi", [{"type": "GREETING", "parameters": {}}], "assistant", "train", "easy", True, "Greeting hi"),
        ("hey", [{"type": "GREETING", "parameters": {}}], "assistant", "train", "easy", True, "Greeting hey"),
        ("good morning", [{"type": "GREETING", "parameters": {}}], "assistant", "train", "easy", True, "Good morning"),
        ("how are you", [{"type": "GREETING", "parameters": {}}], "assistant", "train", "easy", True, "How are you"),
        ("help", [{"type": "HELP", "parameters": {}}], "assistant", "train", "easy", True, "Help command"),
        ("what can you do", [{"type": "HELP", "parameters": {}}], "assistant", "train", "easy", True, "What can you do"),
        ("show commands", [{"type": "HELP", "parameters": {}}], "assistant", "train", "easy", True, "Show commands"),
        ("list commands", [{"type": "HELP", "parameters": {}}], "assistant", "train", "easy", True, "List commands"),
        ("options", [{"type": "HELP", "parameters": {}}], "assistant", "train", "easy", True, "Options query"),
        ("exit", [{"type": "EXIT", "parameters": {}}], "assistant", "train", "easy", True, "Exit command"),
        ("quit", [{"type": "EXIT", "parameters": {}}], "assistant", "train", "easy", True, "Quit command"),
        ("goodbye", [{"type": "EXIT", "parameters": {}}], "assistant", "train", "easy", True, "Goodbye command"),
        ("shutdown system", [{"type": "EXIT", "parameters": {}}], "assistant", "train", "medium", True, "Shutdown system"),
        ("close assistant", [{"type": "EXIT", "parameters": {}}], "assistant", "train", "medium", True, "Close assistant"),
        ("stop listening", [{"type": "STOP_LISTENING", "parameters": {}}], "assistant", "train", "easy", True, "Stop listening"),
        ("mute assistant", [{"type": "STOP_LISTENING", "parameters": {}}], "assistant", "train", "easy", True, "Mute assistant"),
        ("pause listening", [{"type": "STOP_LISTENING", "parameters": {}}], "assistant", "train", "easy", True, "Pause listening"),
        ("go to sleep", [{"type": "STOP_LISTENING", "parameters": {}}], "assistant", "train", "easy", True, "Go to sleep"),

        # Dev
        ("good evening", [{"type": "GREETING", "parameters": {}}], "assistant", "dev", "easy", True, "Good evening"),
        ("greetings", [{"type": "GREETING", "parameters": {}}], "assistant", "dev", "easy", True, "Formal greetings"),
        ("what are your commands", [{"type": "HELP", "parameters": {}}], "assistant", "dev", "medium", True, "What are your commands"),
        ("terminate application", [{"type": "EXIT", "parameters": {}}], "assistant", "dev", "medium", True, "Terminate application"),
        ("shut down the program", [{"type": "EXIT", "parameters": {}}], "assistant", "dev", "medium", True, "Shut down program"),
        ("stop listening to me", [{"type": "STOP_LISTENING", "parameters": {}}], "assistant", "dev", "medium", True, "Stop listening to me"),

        # Test (Held-out)
        ("good afternoon", [{"type": "GREETING", "parameters": {}}], "assistant", "test", "easy", True, "Good afternoon"),
        ("hello assistant", [{"type": "GREETING", "parameters": {}}], "assistant", "test", "medium", True, "Hello with assistant"),
        ("show me available options", [{"type": "HELP", "parameters": {}}], "assistant", "test", "medium", True, "Available options query"),
        ("terminate assistant", [{"type": "EXIT", "parameters": {}}], "assistant", "test", "medium", True, "Terminate assistant"),
        ("pause voice assistant", [{"type": "STOP_LISTENING", "parameters": {}}], "assistant", "test", "medium", True, "Pause voice assistant"),

        # =====================================================================
        # G. Compound Voice Commands (72 items)
        # =====================================================================
        # Train
        ("turn off the ac and headlights", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "medium", True, "Compound object turn off AC and lights"),
        ("turn on the ac and headlights", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "train", "medium", True, "Compound object turn on AC and lights"),
        ("switch off the lights and turn off the ac", [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "medium", True, "Sequential turn off lights and AC"),
        ("activate the headlights and air conditioning", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "train", "medium", True, "Formal activate compound object"),
        ("turn on the lights and switch off the ac", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "medium", True, "Mixed state lights on AC off"),
        ("turn off the headlights but keep the ac on", [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "train", "medium", True, "Contrast clause with keep AC on"),
        ("turn on the ac but turn off the headlights", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "medium", True, "Contrast clause AC on lights off"),
        ("turn off both the ac and the lights", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "medium", True, "Both modifier compound object"),
        ("turn on both headlights and ac", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "train", "medium", True, "Both modifier lights and AC"),
        ("open the driver door and turn on the headlights", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "train", "medium", True, "Door and lights compound"),
        ("close the passenger door and turn off the ac", [{"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "medium", True, "Door and AC compound"),
        ("open both doors and turn on the ac", [{"type": "OPEN_DOOR", "parameters": {"door": "both"}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "train", "medium", True, "Both doors and AC"),
        ("close both doors and switch off the lights", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "medium", True, "Both doors and lights off"),
        ("turn on the headlights, ac, and open the driver door", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_ON", "parameters": {}}, {"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "compound", "train", "hard", True, "Three action compound sequence"),
        ("turn off the ac, switch off the headlights, and close both doors", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}, {"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "compound", "train", "hard", True, "Three action compound shut down"),
        ("turn off the ac first, then turn off the headlights", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "medium", True, "Sequential first then"),
        ("before opening the passenger door, turn off the ac", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "compound", "train", "hard", True, "Sequential inversion with before"),
        ("keep the lights on but turn the ac off", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "medium", True, "Keep lights on contrast"),
        ("close the driver door and turn on the ac, please", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "train", "medium", True, "Compound with trailing politeness"),
        ("turn on the headlights, then turn them off", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "hard", True, "Pronoun anaphora resolution them lights"),
        ("start the ac, then turn it off", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "hard", True, "Pronoun anaphora resolution it AC"),
        ("turn off the ac, not the headlights", [{"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "medium", True, "Negative contrast suppression"),
        ("turn on headlights, not the ac", [{"type": "LIGHTS_ON", "parameters": {}}], "compound", "train", "medium", True, "Negative contrast headlights only"),
        ("open the driver door and open the passenger door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}, {"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "compound", "train", "medium", True, "Two door open compound"),
        ("close the driver door and open the passenger door", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}, {"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "compound", "train", "medium", True, "Opposing door operations"),
        ("open the driver door and turn off the lights", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "medium", True, "Driver door and lights off"),
        ("turn on hazard mode and turn on the headlights", [{"type": "EMERGENCY_ON", "parameters": {}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "train", "medium", True, "Hazard and headlights compound"),
        ("disable emergency mode and turn off the lights", [{"type": "EMERGENCY_OFF", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "medium", True, "Hazard off and lights off"),
        ("activate headlights, then open both doors", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "OPEN_DOOR", "parameters": {"door": "both"}}], "compound", "train", "medium", True, "Activate lights then open both doors"),
        ("close both doors and turn off the ac", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "medium", True, "Close both doors and AC off"),
        ("open the passenger door, then close it", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}, {"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "compound", "train", "hard", True, "Door open and close pronoun"),
        ("switch on the lights, then turn them off", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "train", "hard", True, "Switch lights on and off pronoun"),
        ("turn on the ac and activate hazard lights", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "EMERGENCY_ON", "parameters": {}}], "compound", "train", "medium", True, "AC and hazard compound"),
        ("close the driver door, not the passenger door", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "compound", "train", "medium", True, "Negative contrast door target"),
        ("turn on the lights and check the temperature", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "GET_TEMPERATURE", "parameters": {}}], "compound", "train", "medium", True, "Actuator and sensor query compound"),
        ("close both doors and check the humidity", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}, {"type": "GET_HUMIDITY", "parameters": {}}], "compound", "train", "medium", True, "Door actuator and humidity query"),
        ("turn on the headlights and check the distance sensor", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "GET_DISTANCE", "parameters": {}}], "compound", "train", "medium", True, "Lights and distance query"),
        ("switch off the ac and report the current sensor readings", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "GET_SENSOR_STATUS", "parameters": {}}], "compound", "train", "medium", True, "AC off and diagnostic report"),
        ("open the driver door, turn on the ac, and turn on the headlights", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}, {"type": "TURN_AC_ON", "parameters": {}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "train", "hard", True, "Three action compound door ac lights"),
        ("close both doors, switch off the lights, and turn off the ac", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}, {"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "train", "hard", True, "Three action shut down sequence"),

        # Dev
        ("switch on the headlights and the ac", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "dev", "medium", True, "Compound object lights and ac"),
        ("turn off the headlights and the ac", [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "dev", "medium", True, "Compound object lights off and ac off"),
        ("open the driver door and close the passenger door", [{"type": "OPEN_DOOR", "parameters": {"door": "driver"}}, {"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "compound", "dev", "medium", True, "Dual door operation"),
        ("close the driver door and switch on the lights", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "dev", "medium", True, "Door close lights on"),
        ("open the passenger door and switch on the ac", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "dev", "medium", True, "Passenger door and ac on"),
        ("turn off the headlights, then turn off the ac", [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "dev", "medium", True, "Sequential then lights and ac off"),
        ("before closing both doors, turn off the lights", [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "compound", "dev", "hard", True, "Before inversion doors and lights"),
        ("keep the ac on but switch off the headlights", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "dev", "medium", True, "Contrast ac on lights off"),
        ("leave the headlights off and close the driver door", [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}], "compound", "dev", "medium", True, "Leave lights off close driver door"),
        ("open both doors, turn on the ac, and activate hazard mode", [{"type": "OPEN_DOOR", "parameters": {"door": "both"}}, {"type": "TURN_AC_ON", "parameters": {}}, {"type": "EMERGENCY_ON", "parameters": {}}], "compound", "dev", "hard", True, "Three action compound doors ac hazard"),
        ("turn off the lights, not the ac", [{"type": "LIGHTS_OFF", "parameters": {}}], "compound", "dev", "medium", True, "Negative contrast lights only"),
        ("switch on the ac first, then open the driver door", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "compound", "dev", "medium", True, "Sequential first then AC and door"),

        # Test (Held-out)
        ("power on the headlights and start the air conditioning", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "TURN_AC_ON", "parameters": {}}], "compound", "test", "medium", True, "Synonym verb compound lights and ac"),
        ("shut off the air conditioning and deactivate headlights", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "test", "medium", True, "Synonym shut off and deactivate"),
        ("open the passenger door and turn on the headlights", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "test", "medium", True, "Passenger door and headlights"),
        ("close the driver door and turn off the lights", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}, {"type": "LIGHTS_OFF", "parameters": {}}], "compound", "test", "medium", True, "Close driver and lights off"),
        ("open both doors, then close both doors", [{"type": "OPEN_DOOR", "parameters": {"door": "both"}}, {"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "compound", "test", "medium", True, "Open then close both doors"),
        ("before opening the driver door, turn off the ac", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "OPEN_DOOR", "parameters": {"door": "driver"}}], "compound", "test", "hard", True, "Inverted before order driver door"),
        ("before closing the passenger door, turn on the lights", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "CLOSE_DOOR", "parameters": {"door": "passenger"}}], "compound", "test", "hard", True, "Inverted before order passenger door"),
        ("turn on the air conditioner, then turn it off", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "test", "hard", True, "Pronoun anaphora air conditioner"),
        ("leave the ac off but turn on the headlights", [{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "test", "medium", True, "Contrast leave ac off headlights on"),
        ("keep the headlights on and close both doors", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "compound", "test", "medium", True, "Keep headlights on close doors"),
        ("turn off the headlights, switch off the ac, and close both doors", [{"type": "LIGHTS_OFF", "parameters": {}}, {"type": "TURN_AC_OFF", "parameters": {}}, {"type": "CLOSE_DOOR", "parameters": {"door": "both"}}], "compound", "test", "hard", True, "Three action shut down lights ac doors"),
        ("turn on the ac, open both doors, and turn on the headlights", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "OPEN_DOOR", "parameters": {"door": "both"}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "test", "hard", True, "Three action AC doors headlights"),
        ("turn off the ac, not the lights", [{"type": "TURN_AC_OFF", "parameters": {}}], "compound", "test", "medium", True, "Negative contrast AC not lights"),
        ("open the passenger door, not the driver door", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}], "compound", "test", "medium", True, "Negative contrast passenger only"),
        ("turn on the headlights and check the cabin temperature", [{"type": "LIGHTS_ON", "parameters": {}}, {"type": "GET_TEMPERATURE", "parameters": {}}], "compound", "test", "medium", True, "Lights on and cabin temperature query"),
        ("close both doors and give me the sensor status", [{"type": "CLOSE_DOOR", "parameters": {"door": "both"}}, {"type": "GET_SENSOR_STATUS", "parameters": {}}], "compound", "test", "medium", True, "Close doors and sensor status query"),
        ("enable hazard mode and open both doors", [{"type": "EMERGENCY_ON", "parameters": {}}, {"type": "OPEN_DOOR", "parameters": {"door": "both"}}], "compound", "test", "medium", True, "Hazard on and open doors"),
        ("turn on the ac and turn on the headlights, please", [{"type": "TURN_AC_ON", "parameters": {}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "test", "medium", True, "Compound with trailing please"),
        ("close the driver door first, then turn on the headlights", [{"type": "CLOSE_DOOR", "parameters": {"door": "driver"}}, {"type": "LIGHTS_ON", "parameters": {}}], "compound", "test", "medium", True, "First then sequence driver door and lights"),
        ("open the passenger door first, then turn off the ac", [{"type": "OPEN_DOOR", "parameters": {"door": "passenger"}}, {"type": "TURN_AC_OFF", "parameters": {}}], "compound", "test", "medium", True, "First then sequence passenger door and AC"),

        # =====================================================================
        # H. Unsupported, Out-of-Scope, Ambiguous, and Rejection (40 items)
        # =====================================================================
        # Train
        ("play some rock music", [], "unsupported", "train", "easy", False, "Unsupported media playback"),
        ("play music on spotify", [], "unsupported", "train", "easy", False, "Unsupported streaming service"),
        ("turn on the radio", [], "unsupported", "train", "easy", False, "Unsupported radio feature"),
        ("increase the radio volume", [], "unsupported", "train", "easy", False, "Unsupported audio volume control"),
        ("navigate to downtown", [], "unsupported", "train", "easy", False, "Unsupported GPS navigation"),
        ("set navigation to home", [], "unsupported", "train", "easy", False, "Unsupported home routing"),
        ("what is the fastest route to work", [], "unsupported", "train", "easy", False, "Unsupported traffic route query"),
        ("roll down the driver window", [], "unsupported", "train", "easy", False, "Unsupported window actuator"),
        ("roll up all windows", [], "unsupported", "train", "easy", False, "Unsupported all window actuator"),
        ("open the sunroof", [], "unsupported", "train", "easy", False, "Unsupported sunroof actuator"),
        ("adjust the driver seat forward", [], "unsupported", "train", "easy", False, "Unsupported seat motor"),
        ("set the cabin temperature to 20 degrees", [], "unsupported", "train", "medium", False, "Unsupported temperature adjustment value"),
        ("change temperature to 22 celsius", [], "unsupported", "train", "medium", False, "Unsupported temperature setting value"),
        ("lower the ac temperature", [], "unsupported", "train", "medium", False, "Unsupported thermostat adjustment"),
        ("unlock the driver door", [], "unsupported", "train", "medium", False, "Unsupported lock actuator (unlock != open)"),
        ("lock all vehicle doors", [], "unsupported", "train", "medium", False, "Unsupported lock actuator (lock != close)"),
        ("call mom on the phone", [], "unsupported", "train", "easy", False, "Unsupported cellular call"),
        ("send a text message to john", [], "unsupported", "train", "easy", False, "Unsupported SMS text messaging"),
        ("what is the weather forecast today", [], "unsupported", "train", "easy", False, "Unsupported external weather service"),
        ("turn off the ac and play music", [], "unsupported", "train", "hard", False, "Safety rejection: mixed supported AC and unsupported music"),
        ("open the driver door and navigate home", [], "unsupported", "train", "hard", False, "Safety rejection: mixed supported door and unsupported navigation"),
        ("turn on the headlights and roll down the windows", [], "unsupported", "train", "hard", False, "Safety rejection: mixed supported lights and unsupported windows"),
        ("open the door and turn on the ac", [], "unsupported", "train", "hard", False, "Safety rejection: compound with ambiguous door target"),
        ("close the door and switch off the lights", [], "unsupported", "train", "hard", False, "Safety rejection: compound with ambiguous close door"),

        # Dev
        ("play the latest podcast", [], "unsupported", "dev", "easy", False, "Unsupported podcast player"),
        ("find nearest gas station", [], "unsupported", "dev", "easy", False, "Unsupported POI navigation"),
        ("close the sunroof", [], "unsupported", "dev", "easy", False, "Unsupported sunroof close"),
        ("adjust side mirrors", [], "unsupported", "dev", "easy", False, "Unsupported mirror actuator"),
        ("set ac to 18 degrees", [], "unsupported", "dev", "medium", False, "Unsupported AC temperature value"),
        ("unlock the passenger door", [], "unsupported", "dev", "medium", False, "Unsupported unlock passenger"),
        ("turn on the ac and play jazz", [], "unsupported", "dev", "hard", False, "Safety rejection: mixed supported AC and unsupported jazz"),
        ("open the door and switch on headlights", [], "unsupported", "dev", "hard", False, "Safety rejection: compound ambiguous door target"),

        # Test (Held-out)
        ("play songs by the beatles", [], "unsupported", "test", "easy", False, "Unsupported artist playback"),
        ("navigate to airport", [], "unsupported", "test", "easy", False, "Unsupported airport navigation"),
        ("roll down passenger window", [], "unsupported", "test", "easy", False, "Unsupported passenger window"),
        ("increase temperature by two degrees", [], "unsupported", "test", "medium", False, "Unsupported relative temperature increment"),
        ("unlock all doors", [], "unsupported", "test", "medium", False, "Unsupported unlock all doors"),
        ("read my unread emails", [], "unsupported", "test", "easy", False, "Unsupported email reader"),
        ("turn off the headlights and play the radio", [], "unsupported", "test", "hard", False, "Safety rejection: mixed supported lights and unsupported radio"),
        ("open the door and turn off the ac", [], "unsupported", "test", "hard", False, "Safety rejection: compound ambiguous door target"),
    ]

    records = []
    for idx, (utterance, expected_actions, category, split, difficulty, supported, notes) in enumerate(raw_data, start=1):
        record_id = f"CMD_{idx:03d}"
        normalized = normalize_text(utterance)
        records.append({
            "id": record_id,
            "utterance": utterance,
            "normalized_utterance": normalized,
            "expected_actions": expected_actions,
            "category": category,
            "split": split,
            "difficulty": difficulty,
            "supported": supported,
            "notes": notes,
        })

    return records


def main():
    os.makedirs("data/stt", exist_ok=True)
    records = create_dataset_records()

    jsonl_path = "data/stt/commands.jsonl"
    csv_path = "data/stt/commands.csv"

    # 1. Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 2. Write CSV
    fieldnames = [
        "id",
        "utterance",
        "normalized_utterance",
        "expected_actions",
        "category",
        "split",
        "difficulty",
        "supported",
        "notes",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            csv_row = dict(rec)
            csv_row["expected_actions"] = json.dumps(rec["expected_actions"])
            writer.writerow(csv_row)

    print(f"Generated {len(records)} canonical records.")
    print(f"Saved: {jsonl_path}")
    print(f"Saved: {csv_path}")

    # Summary statistics
    splits = {}
    categories = {}
    supported_counts = {True: 0, False: 0}
    for r in records:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
        categories[r["category"]] = categories.get(r["category"], 0) + 1
        supported_counts[r["supported"]] += 1

    print("\nDataset Split Distribution:")
    for s, c in sorted(splits.items()):
        print(f"  {s:8s}: {c:3d} ({c/len(records)*100:.1f}%)")

    print("\nCategory Distribution:")
    for cat, c in sorted(categories.items()):
        print(f"  {cat:15s}: {c:3d} ({c/len(records)*100:.1f}%)")

    print(f"\nSupported Status: Supported={supported_counts[True]}, Unsupported/Rejected={supported_counts[False]}")


if __name__ == "__main__":
    main()
