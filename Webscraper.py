import json
import httpx

# ushlurl = 'https://ushl.com/ht/#/player/' #followed by a player id starting at 1
URL = 'https://www.eliteprospects.com'
HEADERS = headers = {
"Host": "gql.eliteprospects.com",
"Content-Type": "application/json",
"Accept": "*/*",
"User-Agent": "MyApp/1.0",
"Origin": "https://www.eliteprospects.com",
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
"Priority": "u=1, i",
"Referer": "https://www.eliteprospects.com/"
}
QUERY = '\nquery PlayerStatisticDefault($player: ID, $leagueType: LeagueType, $sort: String) {\n  playerStats(player: $player, statsType: "default,projected", leagueType: $leagueType, sort: $sort) {\n    edges {\n      id\n      player {\n \n detailedPosition \n name\n        position\n        shoots\n        weight{\n      imperial\n      metrics\n    }\n height{\n      imperial\n      metrics\n    }}\n      statsProvider {\n        logoUrl\n        name\n        statsName\n        url\n      }\n      season {\n        slug\n        startYear\n        endYear\n      }\n      statsType\n      team {\n        continent\n        eliteprospectsUrlPath\n        country {\n          eliteprospectsUrlPath\n          flagUrl {\n            small\n          }\n        }\n      }\n      teamName\n      playerRole\n      status\n      contractType\n      league {\n        eliteprospectsUrlPath\n        flagUrl {\n          small\n        }\n      }\n      leagueName\n      regularStats {\n        GP\n        G\n        A\n        PTS\n        PIM\n        PM\n        GAA\n        SVP\n        SVS\n        SO\n        W\n        L\n        T\n        GD\n        GA\n        TOI\n      }\n      postseasonName\n      postseasonType\n      postseasonStats {\n        GP\n        A\n        G\n        PTS\n        PIM\n        PM\n        GAA\n        SVP\n        SVS\n        SO\n        W\n        L\n        T\n        GD\n        GA\n        TOI\n      }\n    }\n  }\n}\n'

VALID_LEAGUES = ["NHL", "OHL", "QMJHL", "USHL", "WHL"]

def sendRequest(playerID):
    eliteProspectsPayloadData = {"operationName":"PlayerStatisticDefault","variables":{"player":playerID,"leagueType":"league","sort":"season"},"query": QUERY}
    with httpx.Client(http2=True, headers=HEADERS) as client:
        return client.post(URL, json=eliteProspectsPayloadData)



def main():
    with open("playerData.txt", mode="x") as file: #TODO change mode to x
        for playerID in [6146, 785522, 577184]: #range(577184, 577185)
            # get data
            response = sendRequest(playerID)
            if response.is_success:
                # parse data
                try:
                    playerData = json.loads(response.text)
                    # print(playerData)
                    seasons = playerData.get("data", {}).get("playerStats", {}).get("edges", {})

                    # get player info
                    playerStats = {}
                    if seasons:
                        player = seasons[0].get("player",{})
                        playerStats["name"] = player.get("name")
                        playerStats["position"] = player.get("position")
                        playerStats["detailedPosition"] = player.get("detailedPosition")
                        playerStats["shoots"] = player.get("shoots")
                        playerStats["height"] = player.get("height", {}).get("metrics", {}) #cm
                        playerStats["weight"] = player.get("weight", {}).get("metrics", {}) #kg
                    
                    # itterate though each season played and collect stats about it
                    playerStats["seasonStats"] = []
                    for season in seasons:
                        team = season.get("teamName")
                        league = season.get("leagueName")
                        status = season.get("status")
                        year = season.get("season", {}).get("slug")

                        # Potentially remove check for valid leagues
                        if season.get("regularStats") and league in VALID_LEAGUES and team:
                            gamesPlayed = season.get("regularStats", {}).get("GP")
                            goals = season.get("regularStats", {}).get("G")
                            assists = season.get("regularStats", {}).get("A")
                            points = season.get("regularStats", {}).get("PTS")
                            penaltyMins = season.get("regularStats", {}).get("PIM")
                            plusMinus = season.get("regularStats", {}).get("PM")
                            seasonStats = {
                                "team": team,
                                "league": league,
                                "status": status,
                                "year": year,
                                "gamesPlayed": gamesPlayed,
                                "goals": goals,
                                "assists": assists,
                                "points": points,
                                "penaltyMins": penaltyMins,
                                "plusMinus": plusMinus
                            }
                            playerStats["seasonStats"].append(seasonStats)
                            # print stats
                            # print(f"team: {team}\nleague: {league}\nstatus: {status}\nyear: {year}")
                            # print(f'''Stats:\n\tGames Played: {gamesPlayed}\n\tGoals: {goals}\n\tAssists: {assists}\n\tPoints: {points}\n\tPenalty Minutes: {penaltyMins}\n\t+\-: {plusMinus}\n''')
                    file.write(json.dumps(playerStats)+"\n")
                except (TypeError, AttributeError) as e:
                    print(e)
            else:
                print(f'''Error with response\n
                        Status Code: {response.status_code}\n
                        Data: {response.text}''')

if __name__ == '__main__':
    main()