"""
FPL Data Loader - Pure Python implementation, NO PANDAS

This module fetches player data from FPL API and processes it using pure Python.
No DataFrame operations, no ambiguous truth value errors.
"""

import logging
import requests

# Import centralized logger
from src.logger import logger

# Static position mapping
POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD"
}


def get_fpl_data() -> list:
    """
    Fetch and process player data from Fantasy Premier League API using pure Python.
    
    NO PANDAS - Pure Python implementation to avoid DataFrame boolean issues.
    
    Returns:
        list: List of player dictionaries with standardized fields
    """
    try:
        logger.info("Fetching FPL data from API (Pure Python)...")
        
        # FPL API endpoint
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json,text/plain,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://fantasy.premierleague.com',
            'Referer': 'https://fantasy.premierleague.com/'
        }
        
        # Make request
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()

        # Extract teams data for team name mapping
        teams_data = data.get('teams', [])
        team_map = {t['id']: t['name'] for t in teams_data}
        logger.info(f"Loaded {len(team_map)} teams: {list(team_map.values())[:5]}...")

        # Extract elements (pure Python list)
        elements = data.get('elements', [])
        if not elements:
            logger.warning("No elements found in API response")
            return []

        logger.info(f"Processing {len(elements)} players with pure Python...")
        
        players = []
        
        for p in elements:
            try:
                # Manual field extraction and validation
                player_id = p.get('id')
                if not player_id:
                    continue
                
                # Extract and validate required fields
                name = str(p.get('web_name', '')).strip()
                if not name:
                    continue
                
                # Team - convert to team name string
                team_id = p.get('team')
                if team_id is None:
                    continue
                try:
                    team_id_int = int(team_id)
                    team = team_map.get(team_id_int, f'Team_{team_id_int}')
                except (ValueError, TypeError):
                    continue
                
                # Position - convert to string format for solver
                element_type = p.get('element_type')
                if element_type is None:
                    continue
                try:
                    position = POSITION_MAP.get(int(element_type), 'Unknown')
                    if position == 'Unknown':
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Price conversion: now_cost / 10.0 (API returns in 1/10 millions)
                now_cost = p.get('now_cost')
                if now_cost is None:
                    continue
                try:
                    price = float(now_cost) / 10.0
                except (ValueError, TypeError):
                    continue
                
                # XP conversion: handle None/NaN cases
                ep_next = p.get('ep_next')
                if ep_next is None or ep_next == '' or str(ep_next).lower() == 'nan':
                    xp = 0.0
                else:
                    try:
                        xp = float(ep_next)
                    except (ValueError, TypeError):
                        xp = 0.0

                # Photo: string field for player image code
                photo = str(p.get('photo', '')).strip()

                # Chance of playing: integer 0-100, default 100 if None
                chance_of_playing_next_round = p.get('chance_of_playing_next_round')
                if chance_of_playing_next_round is None:
                    chance_of_playing = 100
                else:
                    try:
                        chance_of_playing = int(chance_of_playing_next_round)
                        # Ensure it's within valid range
                        chance_of_playing = max(0, min(100, chance_of_playing))
                    except (ValueError, TypeError):
                        chance_of_playing = 100

                # Create standardized player dict
                player_dict = {
                    'id': int(player_id),
                    'name': name,
                    'team': team,
                    'position': position,
                    'price': price,
                    'xp': xp,
                    'photo': photo,
                    'chance_of_playing': chance_of_playing
                }
                
                players.append(player_dict)
                
            except Exception as player_error:
                logger.warning(f"Error processing player {p.get('id', 'unknown')}: {player_error}")
                continue
        
        logger.info(f"SUCCESS: Processed {len(players)} players with pure Python. TYPE: {type(players)}")
        
        # Final validation
        if players:
            sample = players[0]
            logger.debug(f"Sample player: ID={sample['id']}, Name={sample['name']}, Team={sample['team']}, Position={sample['position']}, Price={sample['price']}, XP={sample['xp']}, Photo={sample['photo']}, Chance={sample['chance_of_playing']}")
        
        return players
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching FPL data: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in get_fpl_data: {e}")
        return []
