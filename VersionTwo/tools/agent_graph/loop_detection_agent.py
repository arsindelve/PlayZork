"""
LoopDetectionAgent - Detects when the AI is stuck in unproductive loops.

This agent analyzes raw game history to identify:
- Stuck in same location without progress
- Oscillating between locations (NORTH→SOUTH→NORTH pattern)

When a loop is detected, proposes a breaking action with high confidence.
When no loop, proposes "nothing" with zero confidence.
"""
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from .loop_detection_response import LoopDetectionResponse
from config import GAME_NAME
import logging


class LoopDetectionAgent:
    """
    Agent that detects unproductive loops and proposes breaking actions.

    Always runs in parallel with IssueAgents and ExplorerAgent.
    """

    def __init__(self):
        """Initialize the LoopDetectionAgent"""
        self.proposed_action: str = "nothing"
        self.reason: str = ""
        self.confidence: int = 0
        self.loop_detected: bool = False
        self.loop_type: str = ""
        self.current_location: str = ""

        # Location-change invalidation tracking
        self.last_detection_location: str = ""
        self.last_detection_turn: int = 0

        # Tool call history (for reporting)
        self.tool_calls_history: list = []

    def research_and_propose(
        self,
        research_agent: Runnable,
        decision_llm: BaseChatModel,
        history_tools: list,
        mapper_tools: list,
        current_location: str,
        current_game_response: str,
        current_score: int,
        current_moves: int
    ):
        """
        Analyze raw history for loops and propose breaking action if needed.

        Args:
            research_agent: LLM chain with tools for calling history
            decision_llm: LLM for generating structured proposal
            history_tools: List of available history tools
            mapper_tools: List of available mapper tools
            current_location: Current game location
            current_game_response: Latest game response text
            current_score: Current game score
            current_moves: Current move count
        """
        logger = logging.getLogger(__name__)

        # Store current location for reporting
        self.current_location = current_location

        # If we've moved since last detection, reset loop state
        if self.last_detection_location and self.last_detection_location != current_location:
            logger.info(f"[LoopDetectionAgent] Location changed from '{self.last_detection_location}' to '{current_location}' - previous loop detection invalidated")
            # Reset detection state
            self.loop_detected = False
            self.confidence = 0
            self.proposed_action = "nothing"
            self.loop_type = ""
            self.reason = ""

        logger.info(f"[LoopDetectionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[LoopDetectionAgent] AGENT: LoopDetectionAgent")
        logger.info(f"[LoopDetectionAgent] PURPOSE: Detect stuck/oscillating patterns and break loops")
        logger.info(f"[LoopDetectionAgent] CURRENT LOCATION: {current_location}")
        logger.info(f"[LoopDetectionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[LoopDetectionAgent] Phase 1: Gathering raw history (last 10 turns)")

        # Phase 1: Get raw history (last 10 turns) using history tools
        research_input = {
            "input": "Use get_recent_turns to retrieve the last 10 turns of RAW game history. We need to analyze for stuck/oscillating patterns.",
            "score": current_score,
            "locationName": current_location,
            "moves": current_moves,
            "game_response": current_game_response
        }

        try:
            from llm_utils import invoke_with_retry
            research_response = invoke_with_retry(
                research_agent.with_config(
                    run_name="LoopDetectionAgent Research"
                ),
                research_input,
                operation_name="LoopDetectionAgent Research"
            )
        except Exception as e:
            logger.error(f"[LoopDetectionAgent] Research agent failed: {e}")
            raise

        # Execute tool calls to get raw history
        raw_history = ""
        if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
            tools_map = {tool.name: tool for tool in history_tools}

            logger.info(f"[LoopDetectionAgent] Made {len(research_response.tool_calls)} tool calls:")

            for tool_call in research_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                if tool_name == "get_recent_turns" and tool_name in tools_map:
                    # Force last 10 turns
                    tool_args['n'] = 10
                    logger.info(f"[LoopDetectionAgent]   -> {tool_name}(n={tool_args['n']})")
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    logger.info(f"[LoopDetectionAgent]      Result: {str(tool_result)[:150]}...")
                    raw_history = tool_result

                    # Store tool call history for reporting
                    self.tool_calls_history.append({
                        "tool_name": tool_name,
                        "input": str(tool_args),
                        "output": str(tool_result)
                    })
                    break

        if not raw_history:
            logger.warning("[LoopDetectionAgent] No raw history retrieved")
            self.proposed_action = "nothing"
            self.confidence = 0
            return

        # Phase 2: Get available exits from mapper (needed for both deterministic and LLM)
        logger.info(f"[LoopDetectionAgent] Phase 2: Getting available exits from mapper")

        available_exits = []
        tools_map = {tool.name: tool for tool in mapper_tools}
        if "get_exits_from" in tools_map:
            try:
                logger.info(f"[LoopDetectionAgent]   -> get_exits_from(location='{current_location}')")
                exits_result = tools_map["get_exits_from"].invoke({"location": current_location})
                logger.info(f"[LoopDetectionAgent]      Result: {str(exits_result)[:150]}...")

                # Store tool call history for reporting
                self.tool_calls_history.append({
                    "tool_name": "get_exits_from",
                    "input": str({"location": current_location}),
                    "output": str(exits_result)
                })
                # Parse exits (format: "NORTH → Kitchen, SOUTH → Hallway")
                # CRITICAL: Only include exits that DON'T lead to 'BLOCKED'
                if exits_result and exits_result.strip():
                    for line in exits_result.split(','):
                        parts = line.strip().split('→')
                        if len(parts) >= 2:
                            direction = parts[0].strip()
                            destination = parts[1].strip().strip("'\"")
                            # Only add if destination is NOT blocked
                            if destination.upper() != "BLOCKED":
                                available_exits.append(direction)
                        elif len(parts) == 1:
                            # Fallback: no destination specified, include it
                            direction = parts[0].strip()
                            available_exits.append(direction)
            except Exception as e:
                logger.warning(f"[LoopDetectionAgent] Failed to get exits: {e}")

        logger.info(f"[LoopDetectionAgent] Available exits (excluding BLOCKED): {available_exits}")

        # Phase 2.5: DETERMINISTIC loop detection (check before LLM)
        logger.info(f"[LoopDetectionAgent] Phase 2.5: Running deterministic loop detection checks")

        parsed_turns = self._parse_turns(raw_history)
        logger.info(f"[LoopDetectionAgent] Parsed {len(parsed_turns)} turns from history")

        deterministic_result = self._check_deterministic_loops(
            parsed_turns=parsed_turns,
            current_location=current_location,
            current_score=current_score,
            available_exits=available_exits
        )

        if deterministic_result:
            # Deterministic check found a loop! Skip LLM and return immediately
            logger.info(f"[LoopDetectionAgent] ⚠️ DETERMINISTIC LOOP DETECTED: {deterministic_result['loop_type']}")
            logger.info(f"[LoopDetectionAgent] Evidence: {deterministic_result['evidence']}")

            self.loop_detected = True
            self.loop_type = deterministic_result['loop_type']
            self.proposed_action = deterministic_result['proposed_action']
            self.reason = deterministic_result['reason']
            self.confidence = 98  # Very high confidence for deterministic detection

            # Store location and turn for invalidation tracking
            self.last_detection_location = current_location
            self.last_detection_turn = current_moves

            # Log proposal summary
            logger.info(f"[LoopDetectionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[LoopDetectionAgent] PROPOSAL SUMMARY - ⚠️ DETERMINISTIC LOOP DETECTED")
            logger.info(f"[LoopDetectionAgent] Loop Type: {self.loop_type}")
            logger.info(f"[LoopDetectionAgent] Proposed Action: '{self.proposed_action}' (confidence: {self.confidence}/100)")
            logger.info(f"[LoopDetectionAgent] Reason (first 200 chars): {self.reason[:200]}...")
            logger.info(f"[LoopDetectionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # SKIP LLM analysis - we have deterministic proof of loop
            return

        logger.info(f"[LoopDetectionAgent] No deterministic loop detected, proceeding to LLM analysis")

        # Phase 3: Analyze for loops using LLM (only if deterministic checks didn't find anything)
        logger.info(f"[LoopDetectionAgent] Phase 3: Analyzing for loop patterns with LLM")

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are the LoopDetectionAgent in a {GAME_NAME}-playing AI system.

YOUR RESPONSIBILITY:
Analyze raw game history to detect unproductive loops and propose actions to break them.

LOOP TYPES TO DETECT:

1. **Stuck in Location** (loop_type: "stuck_location")
   - Same location for 5+ CONSECUTIVE turns (not scattered visits!)
   - Example STUCK: [Turn 5: Kitchen, 6: Kitchen, 7: Kitchen, 8: Kitchen, 9: Kitchen]
   - Example NOT STUCK: [Turn 1: Kitchen, 8: Kitchen, 10: Kitchen] = normal exploration
   - No score increase during this time
   - Agent is repeatedly trying different actions at same location, all failing
   - IMPORTANT: Must be CONSECUTIVE - visiting a location multiple times during exploration is NORMAL

2. **Oscillating Between Locations** (loop_type: "oscillating")
   - Moving back and forth between 2-3 locations
   - Example: NORTH → SOUTH → NORTH → SOUTH
   - Example: Kitchen → Hallway → Kitchen → Hallway
   - No meaningful progress being made

3. **Repeated Action at Same Location** (loop_type: "repeated_action")
   - Same command attempted 3+ times AT THE SAME LOCATION
   - Example: "NORTH" failed 3 times while AT "Kitchen"
   - IMPORTANT: "NORTH at Kitchen" is different from "NORTH at Hallway"
   - IMPORTANT: If we've moved to a new location, old repetitions don't count
   - This loop is about context-specific failures, not global command frequency

WHEN LOOP DETECTED:
- Set loop_detected = true
- Set appropriate loop_type
- Propose a RADICALLY DIFFERENT action to break the loop:
  * If location description mentions interactive verbs (climbable, openable, takeable), try that verb!
    Example: "cliff appears climbable" → propose "CLIMB CLIFF" or "CLIMB UP"
    Example: "door can be opened" → propose "OPEN DOOR"
  * Try an unexplored exit from available_exits
  * Try examining objects mentioned in description (EXAMINE item)
  * Try different action categories: if moving failed → interact with objects, if attacking → examine
  * Try INVENTORY to check what you have
- Confidence: 95-100 (very high - loops are bad, must break them!)
- Reason: Explain the loop pattern clearly and why this action breaks it

CRITICAL COMMAND RULES:
- NEVER use semicolons (;) in your proposed action
- NEVER combine multiple commands - propose ONE simple command only
- Use the SIMPLEST possible version of each command
- Examples: 'NORTH', 'EXAMINE DOOR', 'INVENTORY', 'CLIMB CLIFF'
- NOT allowed: 'GO NORTH; EXAMINE ROOM', 'TAKE ITEM AND EXAMINE IT'

WHEN NO LOOP DETECTED:
- Set loop_detected = false
- Set loop_type = ""
- Set proposed_action = "nothing"
- Confidence: 0
- Reason: "No loop pattern detected in recent history"

========================================================
CRITICAL: BE PRECISE, NOT AGGRESSIVE
========================================================

TRUE LOOPS waste turns and prevent progress. But normal exploration is NOT a loop!

REAL LOOP SIGNS (detect these):
+ Same location for 5+ CONSECUTIVE turns (stuck, can't escape)
+ Alternating between just 2 locations repeatedly (A→B→A→B→A)
+ Same action at same location 3+ times with no progress

NOT A LOOP (do NOT flag these):
+ Visiting Kitchen on turns [1, 8, 10] = normal exploration
+ Trying different directions from same hub location
+ Returning to previous locations as part of mapping

Only flag if genuinely STUCK. Scattered visits during exploration are NORMAL.

Respond with structured output."""),
            ("human", """CURRENT LOCATION: {current_location}
CURRENT SCORE: {current_score}

AVAILABLE EXITS (from mapper):
{available_exits}

RAW GAME HISTORY (Last 10 Turns):
{raw_history}

Analyze for loops and propose breaking action if needed:""")
        ])

        analysis_chain = analysis_prompt | decision_llm.with_structured_output(LoopDetectionResponse)

        try:
            from llm_utils import invoke_with_retry
            response = invoke_with_retry(
                analysis_chain.with_config(
                    run_name="LoopDetectionAgent Analysis"
                ),
                {
                    "current_location": current_location,
                    "current_score": current_score,
                    "available_exits": ", ".join(available_exits) if available_exits else "None available",
                    "raw_history": raw_history[:3000]  # Truncate if too long
                },
                operation_name="LoopDetectionAgent Analysis"
            )

            # Store results
            self.loop_detected = response.loop_detected
            self.loop_type = response.loop_type
            self.proposed_action = response.proposed_action
            self.reason = response.reason
            self.confidence = response.confidence

            # Store location and turn for invalidation tracking (if loop detected)
            if self.loop_detected:
                self.last_detection_location = current_location
                self.last_detection_turn = current_moves

            # Log proposal summary
            logger.info(f"[LoopDetectionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            if self.loop_detected:
                logger.info(f"[LoopDetectionAgent] PROPOSAL SUMMARY - ⚠️ LOOP DETECTED")
                logger.info(f"[LoopDetectionAgent] Loop Type: {self.loop_type}")
                logger.info(f"[LoopDetectionAgent] Proposed Action: '{self.proposed_action}' (confidence: {self.confidence}/100)")
                if self.reason:
                    logger.info(f"[LoopDetectionAgent] Reason: {self.reason}")
            else:
                logger.info(f"[LoopDetectionAgent] PROPOSAL SUMMARY - No loop detected")
                logger.info(f"[LoopDetectionAgent] Confidence: 0/100 (no action needed)")
            logger.info(f"[LoopDetectionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        except Exception as e:
            logger.error(f"[LoopDetectionAgent] Analysis failed: {e}")
            raise

    def _parse_turns(self, raw_history: str) -> List[dict]:
        """
        Parse raw history text into structured turn data.

        Args:
            raw_history: Text output from get_recent_turns(n)

        Returns:
            List of dicts with keys: turn_number, location, command, response, score
        """
        turns = []
        lines = raw_history.split('\n')

        current_turn = None
        for line in lines:
            line = line.strip()

            # Match turn header: "Turn #5 (at Forest Path)"
            if line.startswith('Turn #'):
                if current_turn:
                    turns.append(current_turn)

                # Extract turn number and location
                import re
                match = re.match(r'Turn #(\d+) \(at (.+?)\)', line)
                if match:
                    current_turn = {
                        'turn_number': int(match.group(1)),
                        'location': match.group(2),
                        'command': '',
                        'response': '',
                        'score': None
                    }

            # Match command: "  Player: CLIMB TREE"
            elif line.startswith('Player:') and current_turn:
                current_turn['command'] = line.replace('Player:', '').strip()

            # Match response: "  Game: You are now up a tree..."
            elif line.startswith('Game:') and current_turn:
                current_turn['response'] = line.replace('Game:', '').strip()

        # Add last turn
        if current_turn:
            turns.append(current_turn)

        return turns

    def _check_deterministic_loops(
        self,
        parsed_turns: List[dict],
        current_location: str,
        current_score: int,
        available_exits: List[str]
    ) -> Optional[dict]:
        """
        Check for loops using deterministic rules (no LLM needed).

        Returns:
            None if no loop detected, or dict with:
                - loop_detected: True
                - loop_type: str
                - evidence: dict (specific to loop type)
                - proposed_action: str
                - reason: str (very detailed)
        """
        if not parsed_turns:
            return None

        # Check 1: Stuck at Location (Consecutive Turns)
        # CRITICAL: Only detect if stuck for CONSECUTIVE turns, not scattered visits during exploration
        # Example: [1: Kitchen, 8: Kitchen, 10: Kitchen] = normal exploration, NOT stuck
        # Example: [5: Kitchen, 6: Kitchen, 7: Kitchen, 8: Kitchen] = STUCK!

        # Count consecutive turns at current location (from most recent backwards)
        consecutive_at_current = 0
        for turn in reversed(parsed_turns):
            if turn['location'] == current_location:
                consecutive_at_current += 1
            else:
                break  # Hit a different location, stop counting

        # Only flag if stuck at current location for 5+ consecutive turns
        if consecutive_at_current >= 5:
            # Gather evidence for this stuck location
            actions_at_location = [
                turn['command'] for turn in parsed_turns[-consecutive_at_current:]
                if turn['location'] == current_location
            ]

            # Get turn numbers for the consecutive stuck period
            stuck_turn_numbers = [
                turn['turn_number'] for turn in parsed_turns[-consecutive_at_current:]
                if turn['location'] == current_location
            ]

            # Check score stagnation
            scores = [turn.get('score') for turn in parsed_turns if turn.get('score') is not None]
            if not scores:
                scores = [current_score] * len(parsed_turns)
            score_unchanged_turns = 0
            if scores:
                for i in range(len(scores) - 1, 0, -1):
                    if scores[i] == scores[i-1]:
                        score_unchanged_turns += 1
                    else:
                        break

            # Generate breaking action
            breaking_action = self._generate_breaking_action(
                loop_type="stuck_location",
                stuck_location=current_location,
                available_exits=available_exits,
                recent_commands=actions_at_location
            )

            # Generate explicit reason
            reason = self._generate_explicit_reason(
                loop_type="stuck_location",
                evidence={
                    'location': current_location,
                    'visit_turns': stuck_turn_numbers,
                    'actions_attempted': list(set(actions_at_location)),  # unique actions
                    'score_unchanged_turns': score_unchanged_turns,
                    'current_score': current_score,
                    'consecutive_turns': consecutive_at_current
                },
                proposed_action=breaking_action
            )

            return {
                'loop_detected': True,
                'loop_type': 'stuck_location',
                'evidence': {'location': current_location, 'consecutive_turns': consecutive_at_current},
                'proposed_action': breaking_action,
                'reason': reason
            }

        # Check 2: Two-Location Oscillation
        if len(parsed_turns) >= 6:
            recent_locations = [turn['location'] for turn in parsed_turns[-6:]]
            unique_locations = list(set(recent_locations))

            # If only 2 unique locations in last 6 turns = oscillating
            if len(unique_locations) == 2:
                loc_a, loc_b = unique_locations
                # Check if it's alternating pattern
                oscillations = 0
                for i in range(len(recent_locations) - 1):
                    if recent_locations[i] != recent_locations[i+1]:
                        oscillations += 1

                # 3+ location changes in 6 turns = oscillating
                if oscillations >= 3:
                    breaking_action = self._generate_breaking_action(
                        loop_type="oscillating",
                        oscillating_locations=[loc_a, loc_b],
                        available_exits=available_exits,
                        recent_commands=[turn['command'] for turn in parsed_turns[-6:]]
                    )

                    reason = self._generate_explicit_reason(
                        loop_type="oscillating",
                        evidence={
                            'locations': [loc_a, loc_b],
                            'turn_sequence': recent_locations,
                            'oscillation_count': oscillations,
                            'current_score': current_score
                        },
                        proposed_action=breaking_action
                    )

                    return {
                        'loop_detected': True,
                        'loop_type': 'oscillating',
                        'evidence': {'locations': [loc_a, loc_b]},
                        'proposed_action': breaking_action,
                        'reason': reason
                    }

        # Check 3: Repeated Failed Action (Location-Aware)
        if len(parsed_turns) >= 3:
            # Track (location, command) pairs - only flag if SAME action at SAME location
            location_command_pairs = {}
            for turn in parsed_turns[-5:]:
                key = (turn['location'], turn['command'])
                if key not in location_command_pairs:
                    location_command_pairs[key] = []
                location_command_pairs[key].append(turn['turn_number'])

            # Only flag if:
            # 1. Same action attempted 3+ times (increased from 2 to reduce false positives)
            # 2. At SAME location (context-aware)
            # 3. We're CURRENTLY at that location (still relevant)
            for (location, command), turn_numbers in location_command_pairs.items():
                if len(turn_numbers) >= 3:  # Increased threshold from 2 to 3
                    # CRITICAL: Only flag if we're CURRENTLY at this location
                    # If we've moved, the loop is already broken - don't re-detect it
                    if location != current_location:
                        continue  # We've escaped this location - loop already broken

                    # Additional validation: ensure command is not empty
                    if not command or not command.strip():
                        continue

                    # Generate breaking action
                    recent_commands = [turn['command'] for turn in parsed_turns[-5:]]
                    breaking_action = self._generate_breaking_action(
                        loop_type="repeated_action",
                        repeated_command=command,
                        available_exits=available_exits,
                        recent_commands=recent_commands
                    )

                    # Generate explicit reason
                    reason = self._generate_explicit_reason(
                        loop_type="repeated_action",
                        evidence={
                            'command': command,
                            'location': location,
                            'repeat_count': len(turn_numbers),
                            'turn_numbers': turn_numbers,
                            'current_score': current_score
                        },
                        proposed_action=breaking_action
                    )

                    return {
                        'loop_detected': True,
                        'loop_type': 'repeated_action',
                        'evidence': {'command': command, 'location': location, 'count': len(turn_numbers)},
                        'proposed_action': breaking_action,
                        'reason': reason
                    }

        # Check 4: Score Stagnation (Reduced aggressiveness - increased thresholds)
        if len(parsed_turns) >= 8:  # Increased from 5 to 8 turns
            # Check if score hasn't changed (use current_score for all if no score data)
            scores = [current_score] * len(parsed_turns)  # Assume same score
            if all(s == current_score for s in scores[-8:]):  # Check last 8, not 5
                unique_locations = list(set(turn['location'] for turn in parsed_turns[-12:]))  # Wider window (12 vs 8)

                # Only 2 or fewer locations in last 12 turns = REALLY limited exploration (more strict)
                if len(unique_locations) <= 2:  # More strict: only 2 locations, not 3
                    breaking_action = self._generate_breaking_action(
                        loop_type="score_stagnation",
                        available_exits=available_exits,
                        recent_commands=[turn['command'] for turn in parsed_turns[-8:]],  # Use last 8 commands
                        visited_locations=unique_locations
                    )

                    reason = self._generate_explicit_reason(
                        loop_type="score_stagnation",
                        evidence={
                            'stagnant_turns': 8,  # Updated from 5 to 8
                            'locations_visited': unique_locations,
                            'current_score': current_score,
                            'total_turns': len(parsed_turns)
                        },
                        proposed_action=breaking_action
                    )

                    return {
                        'loop_detected': True,
                        'loop_type': 'score_stagnation',
                        'evidence': {'stagnant_turns': 8, 'locations': unique_locations},  # Updated from 5 to 8
                        'proposed_action': breaking_action,
                        'reason': reason
                    }

        return None

    def _generate_breaking_action(
        self,
        loop_type: str,
        available_exits: List[str] = None,
        recent_commands: List[str] = None,
        stuck_location: str = None,
        oscillating_locations: List[str] = None,
        repeated_command: str = None,
        visited_locations: List[str] = None
    ) -> str:
        """
        Generate a breaking action based on loop type.

        Returns a command that will break the detected loop.
        """
        available_exits = available_exits or []
        recent_commands = recent_commands or []

        if loop_type == "stuck_location":
            # Try to move away from stuck location
            # Prefer available exits (these exclude BLOCKED directions)
            for exit_dir in available_exits:
                if exit_dir and exit_dir.upper() not in recent_commands:
                    return exit_dir.upper()
            # If no available exits, try INVENTORY
            return "INVENTORY"

        elif loop_type == "oscillating":
            # Try to escape to a third location
            # Use available exits (these exclude BLOCKED directions)
            for exit_dir in available_exits:
                if exit_dir and exit_dir.upper() not in recent_commands:
                    return exit_dir.upper()
            # If no available exits, try INVENTORY
            return "INVENTORY"

        elif loop_type == "repeated_action":
            # Do something completely different from repeated command
            # Use available exits (these exclude BLOCKED directions)
            if repeated_command:
                # Try available exits that haven't been tried recently
                for exit_dir in available_exits:
                    if exit_dir and exit_dir.upper() not in recent_commands:
                        return exit_dir.upper()

                # Last resort: INVENTORY provides information
                return "INVENTORY"
            return "INVENTORY"

        elif loop_type == "score_stagnation":
            # Need to explore new areas
            # CRITICAL: Don't recommend directions that are already known to be BLOCKED
            # Use available_exits from mapper which excludes blocked directions

            # Try available exits first (these are from mapper, so they exclude BLOCKED)
            if available_exits:
                for exit_dir in available_exits:
                    if exit_dir and exit_dir.upper() not in recent_commands:
                        return exit_dir.upper()

            # If no available exits, try INVENTORY to gather info
            return "INVENTORY"

        return "LOOK"

    def _generate_explicit_reason(
        self,
        loop_type: str,
        evidence: dict,
        proposed_action: str
    ) -> str:
        """
        Generate ultra-detailed reasoning for loop detection.

        The Decision Agent needs to understand:
        1. WHAT loop pattern was detected (with proof)
        2. WHY this is a problem (wasted turns, no progress)
        3. HOW the proposed action fixes it (what's different)
        4. URGENCY (why choose this over IssueAgents)
        """

        if loop_type == "stuck_location":
            location = evidence['location']
            visit_turns = evidence['visit_turns']
            actions = evidence['actions_attempted']
            score_unchanged = evidence['score_unchanged_turns']
            current_score = evidence['current_score']
            consecutive_turns = evidence.get('consecutive_turns', len(visit_turns))

            reason = f"""CRITICAL LOOP DETECTED: Stuck at same location

Evidence:
- Stuck at '{location}' for {consecutive_turns} CONSECUTIVE turns
- Turn numbers: {', '.join(map(str, visit_turns))}
- Actions attempted: {', '.join(actions[:5])}{'...' if len(actions) > 5 else ''}
- Score: {current_score} (unchanged for {score_unchanged} consecutive turns)
- Unable to leave '{location}' or make progress

Pattern Analysis:
- We've been trapped at '{location}' for {consecutive_turns} turns in a row
- Multiple different actions tried, all failing to advance the game
- This location has been thoroughly examined with no progress
- No score increase indicates we're not solving puzzles here
- We're genuinely STUCK, not just passing through during exploration

Proposed Breaking Action: {proposed_action}
- Forces movement to a DIFFERENT location
- Breaks the stuck pattern by trying unexplored directions
- Opens up opportunities for new items, puzzles, and score increases
- Gets us OUT of the deadlock

URGENCY: CRITICAL - Stuck for {consecutive_turns} consecutive turns. Must escape NOW before more turns are wasted. This is blocking all forward progress."""

        elif loop_type == "oscillating":
            locations = evidence['locations']
            turn_sequence = evidence['turn_sequence']
            oscillations = evidence['oscillation_count']
            current_score = evidence['current_score']

            reason = f"""CRITICAL LOOP DETECTED: Oscillating between two locations

Evidence:
- Last 6 turns alternated between: {locations[0]} ↔ {locations[1]}
- Turn sequence: {' → '.join(turn_sequence)}
- Number of back-and-forth cycles: {oscillations}
- Score: {current_score} (no progress while oscillating)

Pattern Analysis:
- We're ping-ponging between just 2 locations with no forward movement
- Neither location is yielding progress or score increases
- We're not exploring any new directions from these locations
- Both locations have been examined multiple times with no new discoveries
- Classic stuck behavior: bouncing back and forth accomplishes nothing

Proposed Breaking Action: {proposed_action}
- Breaks the oscillation pattern by going to a THIRD location
- Escapes the two-location trap entirely
- Explores NEW territory we haven't been stuck in
- Could lead to entirely new area with fresh opportunities

URGENCY: CRITICAL - {oscillations} back-and-forth cycles wasted with ZERO progress. This is textbook stuck behavior that must be broken immediately."""

        elif loop_type == "repeated_action":
            command = evidence['command']
            location = evidence.get('location', 'Unknown')  # Get location if available
            repeat_count = evidence['repeat_count']
            turn_numbers = evidence['turn_numbers']
            current_score = evidence['current_score']

            reason = f"""CRITICAL LOOP DETECTED: Repeated failed action at same location

Evidence:
- Command "{command}" attempted {repeat_count} times AT LOCATION '{location}'
- Repeated on turns: {', '.join(map(str, turn_numbers[:5]))}
- Score: {current_score} (no increase from repeated attempts)
- Same command at same location = same result = no progress

Pattern Analysis:
- We keep trying "{command}" at '{location}' expecting it to work this time
- The game response is clearly the same each time (or action fails)
- Repeating the same failed action at the same location is the definition of a loop
- We haven't found what's needed to make this action succeed here
- Need to do something DIFFERENT to make progress

Proposed Breaking Action: {proposed_action}
- Completely different approach from "{command}"
- Breaks the repetition pattern at '{location}'
- Changes strategy to find what's actually needed
- Productive use of turn instead of repeating failure

URGENCY: HIGH - We've confirmed {repeat_count} times that "{command}" doesn't work at '{location}'. Need to try something else, not keep bashing our head against the same wall."""

        elif loop_type == "score_stagnation":
            stagnant_turns = evidence['stagnant_turns']
            locations = evidence['locations_visited']
            current_score = evidence['current_score']
            total_turns = evidence['total_turns']

            reason = f"""CRITICAL LOOP DETECTED: No progress for extended period

Evidence:
- Score unchanged for last {stagnant_turns} turns: stuck at {current_score} points
- Only {len(locations)} locations explored in last {total_turns} turns: {', '.join(locations)}
- Very limited exploration = not discovering new areas
- No new items acquired, no puzzles solved, no score advancement

Pattern Analysis:
- We're revisiting the same {len(locations)} locations without making progress
- Current strategy (staying in familiar areas) is NOT advancing the game
- Need to break out of this comfortable but unproductive zone
- Goal is {350} points, we're at {current_score} - need NEW areas with NEW treasures
- Stagnation indicates we've exhausted current area's potential

Proposed Breaking Action: {proposed_action}
- Forces exploration into UNEXPLORED territory
- Likely leads to new location (not one of the {len(locations)} we're stuck in)
- New locations = new items, puzzles, score opportunities
- Gets us OUT of the stagnant comfort zone

URGENCY: CRITICAL - {stagnant_turns} turns with ZERO score increase is unacceptable. We're completely stuck and need to force exploration NOW. Every turn in familiar territory is a wasted opportunity."""

        else:
            reason = f"Loop detected: {loop_type}. Proposed action: {proposed_action}"

        return reason
