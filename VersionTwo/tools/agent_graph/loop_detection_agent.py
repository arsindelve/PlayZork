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
            from config import invoke_with_retry
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
                # Parse exits (format: "NORTH → Kitchen, SOUTH → Hallway")
                if exits_result and exits_result.strip():
                    for line in exits_result.split(','):
                        parts = line.strip().split('→')
                        if len(parts) >= 1:
                            direction = parts[0].strip()
                            available_exits.append(direction)
            except Exception as e:
                logger.warning(f"[LoopDetectionAgent] Failed to get exits: {e}")

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
   - Same location for 5+ turns
   - No score increase during this time
   - Agent is repeatedly trying similar failed actions (e.g., GO EAST fails, GO WEST fails)

2. **Oscillating Between Locations** (loop_type: "oscillating")
   - Moving back and forth between 2-3 locations
   - Example: NORTH → SOUTH → NORTH → SOUTH
   - Example: Kitchen → Hallway → Kitchen → Hallway
   - No meaningful progress being made

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

WHEN NO LOOP DETECTED:
- Set loop_detected = false
- Set loop_type = ""
- Set proposed_action = "nothing"
- Confidence: 0
- Reason: "No loop pattern detected in recent history"

========================================================
CRITICAL: BE VERY AGGRESSIVE
========================================================

Loops waste turns and prevent progress. Detect them early!

SIGNS OF LOOPS (detect ANY of these):
+ Same location 5+ turns with no score change
+ Trying different movement commands that all fail
+ Bouncing between 2 locations repeatedly
+ Doing similar actions that don't change the situation

If you see ANY loop pattern, SET loop_detected=true and confidence=95-100.
Don't wait for 5+ turns - catch loops early at turn 5!

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
            from config import invoke_with_retry
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

        # Check 1: Same Location Repetition
        location_visits = {}
        for turn in parsed_turns:
            loc = turn['location']
            if loc not in location_visits:
                location_visits[loc] = []
            location_visits[loc].append(turn['turn_number'])

        # Trigger: Any location visited 3+ times in last 10 turns
        for location, visit_turns in location_visits.items():
            if len(visit_turns) >= 3:
                # Gather evidence
                actions_at_location = [
                    turn['command'] for turn in parsed_turns
                    if turn['location'] == location
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
                    stuck_location=location,
                    available_exits=available_exits,
                    recent_commands=[turn['command'] for turn in parsed_turns[-5:]]
                )

                # Generate explicit reason
                reason = self._generate_explicit_reason(
                    loop_type="stuck_location",
                    evidence={
                        'location': location,
                        'visit_turns': visit_turns,
                        'actions_attempted': list(set(actions_at_location)),  # unique actions
                        'score_unchanged_turns': score_unchanged_turns,
                        'current_score': current_score
                    },
                    proposed_action=breaking_action
                )

                return {
                    'loop_detected': True,
                    'loop_type': 'stuck_location',
                    'evidence': {'location': location, 'visits': len(visit_turns)},
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

        # Check 3: Repeated Failed Action
        if len(parsed_turns) >= 3:
            recent_commands = [turn['command'] for turn in parsed_turns[-5:]]
            command_counts = {}
            for cmd in recent_commands:
                command_counts[cmd] = command_counts.get(cmd, 0) + 1

            # Any command repeated 2+ times in last 5 turns
            for command, count in command_counts.items():
                if count >= 2 and command.strip():
                    # Check if score increased (if so, action is working)
                    scores = [current_score]  # Default if no score data
                    if len(parsed_turns) >= 2:
                        recent_turns = parsed_turns[-5:]
                        turn_nums_with_cmd = [t['turn_number'] for t in recent_turns if t['command'] == command]

                        breaking_action = self._generate_breaking_action(
                            loop_type="repeated_action",
                            repeated_command=command,
                            available_exits=available_exits,
                            recent_commands=recent_commands
                        )

                        reason = self._generate_explicit_reason(
                            loop_type="repeated_action",
                            evidence={
                                'command': command,
                                'repeat_count': count,
                                'turn_numbers': turn_nums_with_cmd,
                                'current_score': current_score
                            },
                            proposed_action=breaking_action
                        )

                        return {
                            'loop_detected': True,
                            'loop_type': 'repeated_action',
                            'evidence': {'command': command, 'count': count},
                            'proposed_action': breaking_action,
                            'reason': reason
                        }

        # Check 4: Score Stagnation
        if len(parsed_turns) >= 5:
            # Check if score hasn't changed (use current_score for all if no score data)
            scores = [current_score] * len(parsed_turns)  # Assume same score
            if all(s == current_score for s in scores[-5:]):
                unique_locations = list(set(turn['location'] for turn in parsed_turns[-8:]))

                # Only 3 or fewer locations in last 8 turns = limited exploration
                if len(unique_locations) <= 3:
                    breaking_action = self._generate_breaking_action(
                        loop_type="score_stagnation",
                        available_exits=available_exits,
                        recent_commands=[turn['command'] for turn in parsed_turns[-5:]],
                        visited_locations=unique_locations
                    )

                    reason = self._generate_explicit_reason(
                        loop_type="score_stagnation",
                        evidence={
                            'stagnant_turns': 5,
                            'locations_visited': unique_locations,
                            'current_score': current_score,
                            'total_turns': len(parsed_turns)
                        },
                        proposed_action=breaking_action
                    )

                    return {
                        'loop_detected': True,
                        'loop_type': 'score_stagnation',
                        'evidence': {'stagnant_turns': 5, 'locations': unique_locations},
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
            # Prefer unexplored exits
            for exit_dir in available_exits:
                if exit_dir:
                    return f"GO {exit_dir}"
            # Fallback: try cardinal directions
            for direction in ["NORTH", "SOUTH", "EAST", "WEST", "UP", "DOWN"]:
                if f"GO {direction}" not in recent_commands:
                    return direction
            return "LOOK"

        elif loop_type == "oscillating":
            # Try to escape to a third location
            # Avoid the two oscillating locations
            for exit_dir in available_exits:
                if exit_dir:
                    return f"GO {exit_dir}"
            # Try a different action category
            for direction in ["EAST", "WEST", "NORTHEAST", "NORTHWEST"]:
                if direction not in recent_commands:
                    return direction
            return "INVENTORY"

        elif loop_type == "repeated_action":
            # Do something completely different from repeated command
            if repeated_command:
                # If repeated command was movement, try interaction
                movement_keywords = ["GO", "NORTH", "SOUTH", "EAST", "WEST", "UP", "DOWN", "CLIMB"]
                is_movement = any(keyword in repeated_command.upper() for keyword in movement_keywords)

                if is_movement:
                    # Try interaction instead
                    return "EXAMINE SURROUNDINGS"
                else:
                    # Try movement instead
                    for exit_dir in available_exits:
                        if exit_dir:
                            return f"GO {exit_dir}"
                    return "NORTH"
            return "LOOK"

        elif loop_type == "score_stagnation":
            # Need to explore new areas
            # Try completely new direction
            all_directions = ["NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST", "UP", "DOWN"]
            for direction in all_directions:
                if direction not in recent_commands and direction not in str(recent_commands):
                    return direction
            # Fallback: try an available exit
            for exit_dir in available_exits:
                if exit_dir:
                    return f"GO {exit_dir}"
            return "NORTH"

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

            reason = f"""CRITICAL LOOP DETECTED: Same location repetition

Evidence:
- Location '{location}' visited on turns: {', '.join(map(str, visit_turns))} ({len(visit_turns)} visits in last 10 turns)
- Actions attempted at this location: {', '.join(actions[:5])}{'...' if len(actions) > 5 else ''}
- Score: {current_score} (unchanged for {score_unchanged} consecutive turns)
- We keep returning to '{location}' expecting different results

Pattern Analysis:
- We're stuck in a cycle involving '{location}'
- This location has been thoroughly examined with no progress
- Every return to this location wastes a turn without advancing the game
- No score increase indicates we're not solving puzzles here
- Definition of insanity: repeating the same location visits expecting different outcomes

Proposed Breaking Action: {proposed_action}
- Abandons '{location}' completely and moves to a NEW area
- Changes strategy from "revisit familiar location" to "explore new territory"
- Opens up opportunities for new items, puzzles, and score increases
- Gets us OUT of the stuck pattern

URGENCY: CRITICAL - We've wasted {len(visit_turns)} turns on this location. Must break out NOW before more turns are lost. This is blocking all forward progress."""

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
            repeat_count = evidence['repeat_count']
            turn_numbers = evidence['turn_numbers']
            current_score = evidence['current_score']

            reason = f"""CRITICAL LOOP DETECTED: Repeated failed action

Evidence:
- Command "{command}" attempted {repeat_count} times in last 5 turns
- Repeated on turns: {', '.join(map(str, turn_numbers[:5]))}
- Score: {current_score} (no increase from repeated attempts)
- Same command = same result = no progress

Pattern Analysis:
- We keep trying "{command}" expecting it to work this time
- The game response is clearly the same each time (or action fails)
- Repeating the same failed action is the definition of a loop
- We haven't found what's needed to make this action succeed
- Need to do something DIFFERENT to make progress

Proposed Breaking Action: {proposed_action}
- Completely different action category from "{command}"
- Stops the futile repetition of a failed command
- Changes strategy to find what's actually needed
- Productive use of turn instead of repeating failure

URGENCY: HIGH - We've confirmed {repeat_count} times that "{command}" doesn't work right now. Need to try something else, not keep bashing our head against the same wall."""

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
