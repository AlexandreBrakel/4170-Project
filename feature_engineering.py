#!/usr/bin/env python3
"""
Feature engineering script for NHL player data.
Creates per-game rate features and normalizes games played.
"""

import json
import sys
import math
from pathlib import Path

"""Safely divide two numbers, returning default if denominator is 0 or None."""
def safe_divide(numerator, denominator, default=0.0):
    if denominator is None or denominator == 0:
        return default
    try:
        return float(numerator or 0) / float(denominator)
    except (TypeError, ValueError):
        return default


"""
    Normalize games played using log transform if needed.
    Returns both original and log-transformed values.
"""
def normalize_games_played(games_played):
    if games_played is None or games_played == 0:
        return {
            'gamesPlayed': 0,
            'gamesPlayed_log': 0.0
        }
    
    games_played = int(games_played)
    # Log transform: log(1 + x) to handle 0 values
    games_played_log = math.log1p(games_played)
    
    return {
        'gamesPlayed': games_played,
        'gamesPlayed_log': float(games_played_log)
    }

"""
    Create per-game rate features for a season.
    Features affected by games played: goals, assists, points, penaltyMins
"""
def create_rate_features(season):
 
    games_played = season.get('gamesPlayed', 0) or 0
    
    # Get normalized games played
    games_features = normalize_games_played(games_played)
    
    rate_features = {
        'goalsPerGame': safe_divide(season.get('goals'), games_played),
        'assistsPerGame': safe_divide(season.get('assists'), games_played),
        'pointsPerGame': safe_divide(season.get('points'), games_played),
        'penaltyMinsPerGame': safe_divide(season.get('penaltyMins'), games_played),
    }
    
    # Plus/minus 
    if games_played > 0:
        rate_features['plusMinusPerGame'] = safe_divide(season.get('plusMinus'), games_played)
    else:
        rate_features['plusMinusPerGame'] = 0.0
    
    return {**games_features, **rate_features}



"""
    Add engineered features to a player's data.
    Modifies the player_data dictionary in place.
"""
def engineer_features(player_data):
   
    for season in player_data.get('seasonStats', []):
        # Create rate features for this season
        new_features = create_rate_features(season)
        
        # Add new features
        season.update(new_features)
    
    return player_data



"""
    Process all players and add engineered features.
"""
def process_data(input_file, output_file):
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    players_processed = 0
    total_seasons = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                player_data = json.loads(line.strip())
                players_processed += 1
                
                # Engineer features
                player_data = engineer_features(player_data)
                
                # Count seasons
                total_seasons += len(player_data.get('seasonStats', []))
                
                # Write updated player data
                outfile.write(json.dumps(player_data, ensure_ascii=False) + "\n")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"{'='*40}")
    print(f"Feature engineering complete!")
    print(f"{'='*40}")
    print(f"Players processed: {players_processed:,}")
    print(f"Total seasons: {total_seasons:,}")
    print(f"\nFeatures added:")
    print(f"  - goalsPerGame: Goals per game")
    print(f"  - assistsPerGame: Assists per game")
    print(f"  - pointsPerGame: Points per game")
    print(f"  - penaltyMinsPerGame: Penalty minutes per game")
    print(f"  - plusMinusPerGame: Plus/minus per game")
    print(f"  - gamesPlayed_log: Log-transformed games played")
    print(f"\nEngineered data saved to: {output_file}")

def main():
    """Main function to run feature engineering."""
    input_file = "playerData_cleaned.txt"
    output_file = "playerData_features.txt"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    process_data(input_file, output_file)

if __name__ == '__main__':
    main()

