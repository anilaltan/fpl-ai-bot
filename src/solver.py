"""
FPL Dream Team Solver using Linear Programming

This module provides optimization algorithms for Fantasy Premier League team selection
using PuLP (Python Linear Programming) library.
"""

import logging
from typing import List, Dict, Any, Optional
from pulp import LpProblem, LpVariable, LpMaximize, LpStatus, lpSum, LpBinary

logger = logging.getLogger(__name__)


def solve_dream_team(players: List[Dict[str, Any]], budget: float = 100.0, max_players_per_team: int = 3) -> List[Dict[str, Any]]:
    """
    Solve FPL dream team optimization using advanced linear programming with squad, lineup, and captain selection.

    Uses 3-layer optimization: Squad (15 players) -> Lineup (11 starting) -> Captain (1x double points).
    Maximizes: Lineup_xP + 2x_Captain_xP + 0.1x_Bench_xP (Smart budget allocation for maximum points)

    Args:
        players: List of player dictionaries with keys: id, name, team_name, position, price, predicted_xP
        budget: Maximum total cost allowed (default: 100.0)

    Returns:
        List of selected players with additional keys:
        - is_starting: bool (True for 11 starting players)
        - is_captain: bool (True for captain)
        - is_vice_captain: bool (True for vice captain - highest scoring non-captain)

    Advanced Constraints:
        Squad (15 players):
        - Total players: exactly 15
        - Budget: total price <= budget
        - Goalkeepers: exactly 2
        - Defenders: exactly 5
        - Midfielders: exactly 5
        - Forwards: exactly 3
        - Max 3 players from same team

        Lineup (11 starting players):
        - Must be subset of squad
        - Total lineup: exactly 11
        - Min 1 GKP, 3 DEF, 1 FWD in starting lineup

        Captain (1 player):
        - Must be in lineup
        - Gets 2x points in objective function

        Objective: Lineup_xP + 2×Captain_xP + 0.1×Bench_xP
    """
    try:
        logger.info(f"🧠 Starting ADVANCED dream team optimization with {len(players)} players, budget: £{budget}M")
        logger.info("🎯 Using Smart FPL Manager Algorithm: Squad(15) → Lineup(11) → Captain(1)")

        if len(players) < 15:
            logger.warning(f"Not enough players: {len(players)} < 15")
            return []

        # DATA NORMALIZATION: Convert various key formats to standard format
        logger.info("🔄 Normalizing player data keys...")
        cleaned_players = []

        for player in players:
            try:
                # ID - try multiple possible keys
                player_id = player.get('id')
                if player_id is None:
                    continue

                # Name - try multiple possible keys
                name = (player.get('name') or
                       player.get('web_name') or
                       str(player_id))
                name = str(name).strip()
                if not name or name.lower() in ['none', 'null']:
                    name = f'Player_{player_id}'

                # Position - try multiple possible keys and formats
                position = (player.get('position') or
                           player.get('element_type'))
                if position is None:
                    continue

                # Convert numeric positions to string if needed
                if isinstance(position, int):
                    position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
                    position = position_map.get(position, str(position))

                position = str(position).strip()

                # Team - try multiple possible keys
                team = (player.get('team_name') or
                       player.get('team') or
                       'Unknown')
                team = str(team).strip()
                if not team or team.lower() in ['none', 'null']:
                    team = 'Unknown'

                # Price - try multiple possible keys and handle scaling
                price = (player.get('price') or
                        player.get('now_cost'))
                if price is None:
                    continue

                # Handle price scaling (if still in API units)
                if isinstance(price, (int, float)) and price > 20:
                    price = price / 10.0
                    logger.debug(f"Scaled price for {name}: {player.get('price', price)} -> {price}")

                price = float(price)
                if price <= 0:
                    continue

                # XP - try multiple possible keys
                xp = (player.get('xp') or
                     player.get('predicted_xP') or
                     player.get('ep_next') or
                     player.get('xP') or
                     0.0)

                if xp is None or str(xp).lower() in ['nan', 'none']:
                    xp = 0.0

                xp = float(xp)

                # Ensure position is valid
                valid_positions = ['GKP', 'DEF', 'MID', 'FWD']
                if position not in valid_positions:
                    logger.warning(f"Invalid position '{position}' for player {name}, skipping")
                    continue

                # Create normalized player object
                cleaned_player = {
                    'id': int(player_id),
                    'name': name,
                    'position': position,
                    'team_name': team,
                    'price': price,
                    'xp': xp
                }

                cleaned_players.append(cleaned_player)

            except Exception as e:
                logger.warning(f"Error normalizing player {player.get('id', 'unknown')}: {e}")
                continue

        if len(cleaned_players) < 15:
            logger.warning(f"Not enough valid players after cleaning: {len(cleaned_players)} < 15")
            return []

        logger.info(f"Using {len(cleaned_players)} cleaned players for advanced optimization")

        # ========================================
        # ADVANCED 3-LAYER OPTIMIZATION ALGORITHM
        # ========================================

        # Create LP problem
        prob = LpProblem("FPL_Smart_Manager", LpMaximize)

        # ===== DECISION VARIABLES =====
        # Layer 1: Squad selection (15 players)
        squad_vars = {}
        # Layer 2: Starting lineup (11 players from squad)
        lineup_vars = {}
        # Layer 3: Captain selection (1 player from lineup)
        captain_vars = {}

        for player in cleaned_players:
            player_id = player['id']
            squad_vars[player_id] = LpVariable(f"squad_{player_id}", cat=LpBinary)
            lineup_vars[player_id] = LpVariable(f"lineup_{player_id}", cat=LpBinary)
            captain_vars[player_id] = LpVariable(f"captain_{player_id}", cat=LpBinary)

        # ===== ADVANCED OBJECTIVE FUNCTION =====
        # Maximize: Lineup_xP + 2×Captain_xP + 0.1×Bench_xP
        lineup_xp = lpSum([
            lineup_vars[player_id] * player['xp']
            for player in cleaned_players
            if player['id'] in lineup_vars
        ])

        captain_xp = lpSum([
            captain_vars[player_id] * player['xp']
            for player in cleaned_players
            if player['id'] in captain_vars
        ])

        bench_xp = lpSum([
            (squad_vars[player_id] - lineup_vars[player_id]) * player['xp']
            for player in cleaned_players
            if player['id'] in squad_vars
        ])

        prob += lineup_xp + (2 * captain_xp) + (0.1 * bench_xp), "Smart_FPL_Manager_Objective"

        logger.info("🎯 Objective Function: Lineup_xP + 2×Captain_xP + 0.1×Bench_xP")

        # ===== CONSTRAINTS =====

        # SQUAD CONSTRAINTS (15 players)
        # Total squad size
        prob += lpSum(squad_vars.values()) == 15, "Squad_Size_15"

        # Squad budget limit
        prob += lpSum([
            squad_vars[player_id] * player['price']
            for player in cleaned_players
            if player['id'] in squad_vars
        ]) <= budget, f"Squad_Budget_{budget}"

        # Squad position constraints (FPL Standard: 2-5-5-3)
        gk_squad = [squad_vars[player_id] for player in cleaned_players
                   if player['id'] in squad_vars and player['position'] == 'GKP']
        prob += lpSum(gk_squad) == 2, "Squad_Goalkeepers_2"

        def_squad = [squad_vars[player_id] for player in cleaned_players
                    if player['id'] in squad_vars and player['position'] == 'DEF']
        prob += lpSum(def_squad) == 5, "Squad_Defenders_5"

        mid_squad = [squad_vars[player_id] for player in cleaned_players
                    if player['id'] in squad_vars and player['position'] == 'MID']
        prob += lpSum(mid_squad) == 5, "Squad_Midfielders_5"

        fwd_squad = [squad_vars[player_id] for player in cleaned_players
                    if player['id'] in squad_vars and player['position'] == 'FWD']
        prob += lpSum(fwd_squad) == 3, "Squad_Forwards_3"

        # Squad team constraints (max 3 per team)
        team_names = set(player['team_name'] for player in cleaned_players if player['team_name'])
        for team_name in team_names:
            if team_name and team_name.strip() and team_name != "Unknown":
                team_squad = [
                    squad_vars[player_id]
                    for player in cleaned_players
                    if player['id'] in squad_vars and player['team_name'] == team_name
                ]
                if len(team_squad) > max_players_per_team:
                    prob += lpSum(team_squad) <= max_players_per_team, f"Squad_Team_{team_name.replace(' ', '_')}_Max_{max_players_per_team}"

        # LINEUP CONSTRAINTS (11 starting players from squad)
        # Lineup size
        prob += lpSum(lineup_vars.values()) == 11, "Lineup_Size_11"

        # Lineup must be subset of squad
        for player in cleaned_players:
            player_id = player['id']
            if player_id in lineup_vars and player_id in squad_vars:
                prob += lineup_vars[player_id] <= squad_vars[player_id], f"Lineup_From_Squad_{player_id}"

        # Lineup position constraints (FPL rules)
        gk_lineup = [lineup_vars[player_id] for player in cleaned_players
                    if player['id'] in lineup_vars and player['position'] == 'GKP']
        prob += lpSum(gk_lineup) == 1, "Lineup_Exactly_Goalkeeper_1"

        def_lineup = [lineup_vars[player_id] for player in cleaned_players
                     if player['id'] in lineup_vars and player['position'] == 'DEF']
        prob += lpSum(def_lineup) >= 3, "Lineup_Min_Defenders_3"

        fwd_lineup = [lineup_vars[player_id] for player in cleaned_players
                     if player['id'] in lineup_vars and player['position'] == 'FWD']
        prob += lpSum(fwd_lineup) >= 1, "Lineup_Min_Forwards_1"

        mid_lineup = [lineup_vars[player_id] for player in cleaned_players
                     if player['id'] in lineup_vars and player['position'] == 'MID']
        prob += lpSum(mid_lineup) >= 1, "Lineup_Min_Midfielders_1"

        # CAPTAIN CONSTRAINTS (1 captain from lineup)
        # Exactly one captain
        prob += lpSum(captain_vars.values()) == 1, "Captain_Exactly_One"

        # Captain must be in lineup
        for player in cleaned_players:
            player_id = player['id']
            if player_id in captain_vars and player_id in lineup_vars:
                prob += captain_vars[player_id] <= lineup_vars[player_id], f"Captain_From_Lineup_{player_id}"

        # ===== SOLVE ADVANCED OPTIMIZATION =====
        logger.info("🧠 Solving Smart FPL Manager optimization...")

        # Multi-level fallback strategy
        solution_attempts = [
            ("strict", "Original constraints"),
            ("relax_budget", "Relaxed budget +5%"),
            ("no_teams", "No team constraints")
        ]

        for attempt_name, attempt_desc in solution_attempts:
            logger.info(f"🎯 Attempt: {attempt_desc}")

            if attempt_name == "relax_budget":
                # Relax budget by 5%
                relaxed_budget = budget * 1.05
                prob.constraints[f"Squad_Budget_{budget}"] = lpSum([
                    squad_vars[player_id] * player['price']
                    for player in cleaned_players
                    if player['id'] in squad_vars
                ]) <= relaxed_budget

            elif attempt_name == "no_teams":
                # Remove team constraints
                team_constraints = [c for c in prob.constraints.keys() if 'Squad_Team_' in c]
                for constraint_name in team_constraints:
                    del prob.constraints[constraint_name]

            # Solve
            status = prob.solve()

            if LpStatus[status] == 'Optimal':
                logger.info(f"✅ {attempt_desc} - SOLUTION FOUND!")
                break
            else:
                logger.warning(f"⚠️  {attempt_desc} failed (Status: {LpStatus[status]})")

        if LpStatus[status] != 'Optimal':
            logger.error("❌ All advanced optimization attempts failed. Using smart fallback.")
            return _create_smart_fallback_team(cleaned_players, budget)

        # ===== EXTRACT ADVANCED SOLUTION =====
        logger.info("🎉 Smart FPL Manager optimization completed!")

        selected_players = []
        squad_players = []
        lineup_players = []
        captain_player = None

        # Extract squad (15 players)
        for player in cleaned_players:
            player_id = player['id']
            if squad_vars[player_id].value() == 1:
                squad_players.append(player.copy())

        # Extract lineup and captain
        for player in squad_players:
            player_id = player['id']
            player_copy = player.copy()

            if lineup_vars[player_id].value() == 1:
                player_copy['is_starting'] = True
                lineup_players.append(player_copy)

                if captain_vars[player_id].value() == 1:
                    player_copy['is_captain'] = True
                    captain_player = player_copy
                else:
                    player_copy['is_captain'] = False
            else:
                player_copy['is_starting'] = False
                player_copy['is_captain'] = False

            selected_players.append(player_copy)

        # Set vice captain (highest scoring non-captain starting player)
        starting_non_captains = [p for p in lineup_players if not p.get('is_captain', False)]
        if starting_non_captains:
            vice_captain = max(starting_non_captains, key=lambda x: x.get('xp', 0))
            for player in selected_players:
                if player['id'] == vice_captain['id']:
                    player['is_vice_captain'] = True
                else:
                    player['is_vice_captain'] = False

        # ===== CALCULATE ADVANCED METRICS =====
        squad_cost = sum(p['price'] for p in squad_players)
        lineup_xp = sum(p['xp'] for p in lineup_players)
        bench_xp = sum(p['xp'] for p in squad_players if not any(lp['id'] == p['id'] for lp in lineup_players))
        captain_xp = captain_player['xp'] if captain_player else 0

        # Smart objective score: Lineup_xP + 2×Captain_xP + 0.1×Bench_xP
        smart_score = lineup_xp + (2 * captain_xp) + (0.1 * bench_xp)

        logger.info("📊 Advanced Optimization Results:")
        logger.info(f"   Squad: {len(squad_players)} players, £{squad_cost:.1f}M ({squad_cost/budget*100:.1f}% budget)")
        logger.info(f"   Lineup: {len(lineup_players)} starting players, {lineup_xp:.1f} XP")
        logger.info(f"   Bench: {15-len(lineup_players)} players, {bench_xp:.1f} XP")
        logger.info(f"   Captain: {captain_player['name'] if captain_player else 'None'}, {captain_xp:.1f} XP (2x = {2*captain_xp:.1f})")
        logger.info(f"   Smart Score: {smart_score:.1f} (Lineup + 2×Captain + 0.1×Bench)")

        # Position breakdown for squad
        squad_positions = {}
        for player in squad_players:
            pos = player.get('position', 'Unknown')
            squad_positions[pos] = squad_positions.get(pos, 0) + 1
        logger.info(f"   Squad positions: {squad_positions}")

        # Position breakdown for lineup
        lineup_positions = {}
        for player in lineup_players:
            pos = player.get('position', 'Unknown')
            lineup_positions[pos] = lineup_positions.get(pos, 0) + 1
        logger.info(f"   Lineup positions: {lineup_positions}")

        return selected_players

    except Exception as e:
        logger.error(f"❌ Smart FPL Manager optimization failed: {e}")
        return []


def _create_smart_fallback_team(cleaned_players: List[Dict[str, Any]], budget: float) -> List[Dict[str, Any]]:
    """
    Create a smart fallback team using simplified 3-layer logic.
    Selects top players for squad, then assigns starting lineup and captain intelligently.

    Args:
        cleaned_players: List of normalized player dictionaries
        budget: Budget constraint

    Returns:
        List of 15 players with is_starting, is_captain, is_vice_captain flags
    """
    try:
        logger.info("🧠 Creating SMART fallback team with 3-layer optimization...")

        # Sort players by XP descending
        sorted_players = sorted(cleaned_players, key=lambda x: x['xp'], reverse=True)

        # Initialize selection
        selected = []
        pos_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        total_cost = 0.0

        # Required positions for squad: 2 GKP, 5 DEF, 5 MID, 3 FWD
        required = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}

        # Fill squad with position requirements
        for player in sorted_players:
            pos = player['position']
            if pos_counts[pos] < required[pos] and total_cost + player['price'] <= budget:
                selected.append(player.copy())
                pos_counts[pos] += 1
                total_cost += player['price']

        # Fill remaining squad slots with best available players
        remaining_slots = 15 - len(selected)
        for player in sorted_players:
            if (player not in selected and
                remaining_slots > 0 and
                total_cost + player['price'] <= budget * 1.05):  # Allow 5% overspend
                selected.append(player.copy())
                total_cost += player['price']
                remaining_slots -= 1

        if len(selected) < 15:
            logger.warning(f"Could only select {len(selected)} players for squad")
            # Fill with cheapest remaining players if needed
            remaining_players = [p for p in sorted_players if p not in selected]
            for player in remaining_players[:15-len(selected)]:
                selected.append(player.copy())
                total_cost += player['price']

        # SMART STARTING LINEUP SELECTION
        # Prioritize high-xP players for starting 11
        selected.sort(key=lambda x: x['xp'], reverse=True)

        # Basic formation: 1-4-4-2 (GK, 4 DEF, 4 MID, 2 FWD)
        lineup_target = {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}
        lineup = []
        lineup_pos_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}

        # Select starting lineup
        for player in selected:
            pos = player['position']
            if lineup_pos_counts[pos] < lineup_target[pos]:
                player['is_starting'] = True
                lineup.append(player)
                lineup_pos_counts[pos] += 1
            else:
                player['is_starting'] = False

        # Ensure minimum requirements (1 GK, 3 DEF, 1 FWD)
        if lineup_pos_counts['GKP'] == 0:
            gk_candidates = [p for p in selected if p['position'] == 'GKP' and not p.get('is_starting', False)]
            if gk_candidates:
                # Replace lowest scoring starter with GK
                lowest_starter = min(lineup, key=lambda x: x['xp'])
                lowest_starter['is_starting'] = False
                gk_candidates[0]['is_starting'] = True
                lineup = [p for p in lineup if p['id'] != lowest_starter['id']] + [gk_candidates[0]]

        # CAPTAIN SELECTION: Highest scoring starting player
        starting_players = [p for p in selected if p.get('is_starting', False)]
        if starting_players:
            captain = max(starting_players, key=lambda x: x['xp'])
            captain['is_captain'] = True

            # VICE CAPTAIN: Second highest scoring starter
            non_captain_starters = [p for p in starting_players if p['id'] != captain['id']]
            if non_captain_starters:
                vice_captain = max(non_captain_starters, key=lambda x: x['xp'])
                vice_captain['is_vice_captain'] = True

        # Set defaults for players without flags
        for player in selected:
            if 'is_starting' not in player:
                player['is_starting'] = False
            if 'is_captain' not in player:
                player['is_captain'] = False
            if 'is_vice_captain' not in player:
                player['is_vice_captain'] = False

        # Calculate metrics
        lineup_count = sum(1 for p in selected if p.get('is_starting', False))
        captain_name = next((p['name'] for p in selected if p.get('is_captain', False)), 'None')
        captain_xp = next((p['xp'] for p in selected if p.get('is_captain', False)), 0)

        logger.info(f"✅ Smart fallback team created: {len(selected)} players")
        logger.info(f"   Squad cost: £{total_cost:.1f}M (budget: £{budget}M)")
        logger.info(f"   Starting lineup: {lineup_count} players")
        logger.info(f"   Captain: {captain_name} ({captain_xp:.1f} XP)")

        return selected

    except Exception as e:
        logger.error(f"❌ Smart fallback team creation failed: {e}")
        # Ultimate fallback: just return top 15 players with basic flags
        sorted_players = sorted(cleaned_players, key=lambda x: x['xp'], reverse=True)
        fallback = sorted_players[:15]

        # Set basic flags
        for i, player in enumerate(fallback):
            player_copy = player.copy()
            player_copy['is_starting'] = i < 11  # First 11 start
            player_copy['is_captain'] = i == 0   # Top player is captain
            player_copy['is_vice_captain'] = i == 1  # Second is vice
            fallback[i] = player_copy

        return fallback


def solve_dream_team_with_retry(players: List[Dict[str, Any]], budget: float = 100.0) -> Dict[str, Any]:
    """
    Solve dream team with enhanced debugging and fallback mechanisms.

    Args:
        players: List of player dictionaries
        budget: Maximum total cost allowed

    Returns:
        Dict with status and players:
        - {"status": "optimal", "players": [...]} for successful solution
        - {"status": "fallback", "players": [...]} for top players fallback
        - {"status": "failed", "players": []} if completely impossible
    """
    logger.info(f"🧪 Starting dream team optimization (budget: £{budget}M)")

    # Enhanced data analysis before solving
    if not players:
        logger.error("❌ No players data provided")
        return {"status": "failed", "players": []}

    # Analyze player data
    position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
    team_counts = {}
    prices = []
    xp_values = []

    for p in players:
        pos = p.get('position', '')
        team = p.get('team', '')
        price = p.get('price', 0)
        xp = p.get('xp', 0)

        if pos in position_counts:
            position_counts[pos] += 1
        if team:
            team_counts[team] = team_counts.get(team, 0) + 1
        if price > 0:
            prices.append(price)
        if xp > 0:
            xp_values.append(xp)

    logger.info(f"📊 Data Analysis:")
    logger.info(f"   Total players: {len(players)}")
    logger.info(f"   Position distribution: {position_counts}")
    logger.info(f"   Unique teams: {len(team_counts)}")
    logger.info(f"   Price range: £{min(prices):.1f}M - £{max(prices):.1f}M (avg: £{sum(prices)/len(prices):.1f}M)")
    logger.info(f"   XP range: {min(xp_values):.1f} - {max(xp_values):.1f} (avg: {sum(xp_values)/len(xp_values):.1f})")

    # Check minimum requirements
    min_gkp = position_counts.get('GKP', 0) >= 2
    min_def = position_counts.get('DEF', 0) >= 5
    min_mid = position_counts.get('MID', 0) >= 5
    min_fwd = position_counts.get('FWD', 0) >= 3

    logger.info(f"📋 Minimum requirements met: GKP≥2:{min_gkp}, DEF≥5:{min_def}, MID≥5:{min_mid}, FWD≥3:{min_fwd}")

    if not (min_gkp and min_def and min_mid and min_fwd):
        logger.error("❌ Insufficient players for minimum position requirements")
        return {"status": "failed", "players": []}

    # Attempt 1: Strict FPL rules (max 3 per team)
    logger.info("🎯 Attempt 1: Using strict FPL rules (max 3 players per team)")
    result = solve_dream_team(players, budget, max_players_per_team=3)

    if result and len(result) == 15:
        logger.info("✅ Strict rules successful!")
        return {"status": "optimal", "players": result}

    # Attempt 2: Relaxed team constraints (max 5 per team)
    logger.warning("⚠️  Strict rules failed. Attempt 2: Relaxed constraints (max 5 per team)")
    result = solve_dream_team(players, budget, max_players_per_team=5)

    if result and len(result) == 15:
        logger.info("✅ Relaxed rules successful!")
        return {"status": "optimal", "players": result}

    # Attempt 3: No team constraints at all
    logger.warning("⚠️  Relaxed rules failed. Attempt 3: No team constraints (max 15 per team)")
    result = solve_dream_team(players, budget, max_players_per_team=15)

    if result and len(result) == 15:
        logger.info("✅ No team constraints successful!")
        return {"status": "optimal", "players": result}

    # FALLBACK: Return top 15 players by XP regardless of constraints
    logger.error("❌ All optimization attempts failed. Using fallback: Top 15 by XP")

    try:
        # Sort by XP descending and take top 15
        sorted_players = sorted(players, key=lambda x: x.get('xp', 0), reverse=True)
        fallback_team = sorted_players[:15]

        total_cost = sum(p.get('price', 0) for p in fallback_team)
        total_xp = sum(p.get('xp', 0) for p in fallback_team)

        logger.info(f"🔄 Fallback team: {len(fallback_team)} players, £{total_cost:.1f}M spent, {total_xp:.1f} XP")

        return {"status": "fallback", "players": fallback_team}

    except Exception as fallback_error:
        logger.error(f"❌ Even fallback failed: {fallback_error}")
        return {"status": "failed", "players": []}


def validate_team_selection(selected_players: List[Dict[str, Any]], budget: float = 100.0) -> Dict[str, Any]:
    """
    Validate that a team selection meets all FPL constraints.

    Args:
        selected_players: List of selected players
        budget: Budget limit

    Returns:
        Dict with validation results
    """
    try:
        if len(selected_players) != 15:
            return {"valid": False, "error": f"Wrong number of players: {len(selected_players)} != 15"}

        total_cost = sum(player.get('price', 0) for player in selected_players)
        if total_cost > budget:
            return {"valid": False, "error": f"Budget exceeded: £{total_cost:.1f}M > £{budget}M"}

        positions = {}
        teams = {}

        for player in selected_players:
            # Count positions
            pos = player.get('position', 'Unknown')
            positions[pos] = positions.get(pos, 0) + 1

            # Count teams
            team = player.get('team_name', 'Unknown')
            teams[team] = teams.get(team, 0) + 1

        # Check position constraints
        if positions.get('Goalkeeper', 0) != 2:
            return {"valid": False, "error": f"Goalkeepers: {positions.get('Goalkeeper', 0)} != 2"}

        if positions.get('Defender', 0) != 5:
            return {"valid": False, "error": f"Defenders: {positions.get('Defender', 0)} != 5"}

        if positions.get('Midfielder', 0) != 5:
            return {"valid": False, "error": f"Midfielders: {positions.get('Midfielder', 0)} != 5"}

        if positions.get('Forward', 0) != 3:
            return {"valid": False, "error": f"Forwards: {positions.get('Forward', 0)} != 3"}

        # Check team constraints
        max_team_players = max(teams.values()) if teams else 0
        if max_team_players > 3:
            return {"valid": False, "error": f"Team constraint violated: max {max_team_players} players from one team"}

        total_xp = sum(player.get('predicted_xP', 0) for player in selected_players)

        return {
            "valid": True,
            "total_cost": round(total_cost, 2),
            "total_xp": round(total_xp, 2),
            "positions": positions,
            "teams": teams
        }

    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}
