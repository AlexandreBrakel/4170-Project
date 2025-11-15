#!/usr/bin/env python3
"""
Data cleaning script for NHL player data.
Removes seasons from leagues that are not in the valid list and drops
players who have not reached the minimum NHL games .
"""

import json
import sys
from pathlib import Path

# valid leagues to keep
VALID_LEAGUES = ["NHL", "OHL", "QMJHL", "USHL", "WHL", "NCAA", "AHL"]

# minNHL games required 
MIN_NHL_GAMES = 30

def clean_player_data(
    input_file,
    output_file,
    valid_leagues=None,
    min_nhl_games=MIN_NHL_GAMES,
):
    if valid_leagues is None:
        valid_leagues = VALID_LEAGUES
    
    # set lookup
    valid_leagues_set = set(valid_leagues)
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"error: input file not found.")
        sys.exit(1)
    
    players_processed = 0
    players_kept = 0
    players_removed_no_valid_league = 0
    players_removed_low_nhl_games = 0
    total_seasons_before = 0
    total_seasons_after = 0
    
    print(f"Cleaning data from '{input_file}'...")
    print(f"Keeping seasons from leagues: {', '.join(valid_leagues)}")
    print(f"Minimum NHL games required: {min_nhl_games}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                player_data = json.loads(line.strip())
                players_processed += 1
                
                # get original  stats
                original_seasons = player_data.get("seasonStats", [])
                total_seasons_before += len(original_seasons)
                
                # only valid leagues and seasons with games played
                filtered_seasons = [
                    season for season in original_seasons
                    if season.get("league") in valid_leagues_set and season.get("gamesPlayed") and season.get("gamesPlayed") > 0
                ]
                
                total_nhl_games = sum(
                    int(season.get("gamesPlayed") or 0)
                    for season in filtered_seasons
                    if season.get("league") == "NHL"
                )
                
                # Update player data with filtered seasons
                player_data["seasonStats"] = filtered_seasons
                
                # Only keep players with at least one valid league and min nhl games
                if not filtered_seasons:
                    players_removed_no_valid_league += 1
                    continue

                if total_nhl_games < min_nhl_games:
                    players_removed_low_nhl_games += 1
                    continue

                total_seasons_after += len(filtered_seasons)
                outfile.write(json.dumps(player_data, ensure_ascii=False) + "\n")
                players_kept += 1
             
                if players_processed % 500 == 0:
                    print(f"Processed {players_processed} players...", end='\r')
                    
            except json.JSONDecodeError as e:
                print(f"\nWarning: Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"\nWarning: Error processing line {line_num}: {e}")
                continue
    
    print(f"\n{'='*40}")
    print(f"Cleaning complete!")
    print(f"{'='*40}")
    print(f"Players processed: {players_processed}")
    print(f"Players kept: {players_kept}")
    print(
        f"Players removed (no valid seasons): {players_removed_no_valid_league}"
    )
    print(
        f"Players removed (fewer than {min_nhl_games} NHL games): "
        f"{players_removed_low_nhl_games}"
    )
    print(f"Total seasons before: {total_seasons_before}")
    print(f"Total seasons after: {total_seasons_after}")
    print(f"Seasons removed: {total_seasons_before - total_seasons_after}")
    print(f"\nCleaned data saved to: {output_file}")


def main():
    # file paths
    input_file = "playerData.txt"
    output_file = "playerData_cleaned.txt"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # optional cunstoms leagues
    valid_leagues = VALID_LEAGUES
    if len(sys.argv) > 3:
        valid_leagues = [league.strip() for league in sys.argv[3].split(",")]
        print(f"Using custom valid leagues: {valid_leagues}")

    min_nhl_games = MIN_NHL_GAMES
    if len(sys.argv) > 4:
        try:
            min_nhl_games = int(sys.argv[4])
        except ValueError:
            print(
                f"Warning: invalid min nhl games'{sys.argv[4]}'. "
                f"Using default value of {MIN_NHL_GAMES}."
            )
            min_nhl_games = MIN_NHL_GAMES
    
    clean_player_data(input_file, output_file, valid_leagues, min_nhl_games)


if __name__ == '__main__':
    main()

