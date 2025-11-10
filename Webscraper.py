import json
import httpx
import sys

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
QUERY = '\nquery PlayerStatisticDefault($player: ID, $leagueType: LeagueType, $sort: String) {\n  playerStats(player: $player, statsType: "default,projected", leagueType: $leagueType, sort: $sort, playerHasPlayedInLeague: "nhl") {\n    edges {\n      id\n      player {\n \n detailedPosition \n name\n        position\n        shoots\n        weight{\n      imperial\n      metrics\n    }\n height{\n      imperial\n      metrics\n    }} season {\n        slug\n        startYear\n        endYear\n      }\n      teamName\n     status\n leagueName\n      regularStats {\n        GP\n        G\n        A\n        PTS\n        PIM\n        PM\n        GAA\n        SVP\n        SVS\n        SO\n        W\n        L\n        T\n        GD\n        GA\n        TOI\n      }\n      postseasonName\n      postseasonType\n      postseasonStats {\n        GP\n        A\n        G\n        PTS\n        PIM\n        PM\n        GAA\n        SVP\n        SVS\n        SO\n        W\n        L\n        T\n        GD\n        GA\n        TOI\n      }\n    }\n  }\n}\n'


SecondQuery = 'query Players($limit: Int, $sort: String, $offset: Int) {\n  players(offset: $offset limit: $limit hasPlayedInLeague: "nhl", sort: $sort) {\n    edges {\n      name\n   id     }}}'


# VALID_LEAGUES = ["NHL", "OHL", "QMJHL", "USHL", "WHL"]

def sendRequest(playerID):
    eliteProspectsPayloadData = {"operationName":"PlayerStatisticDefault","variables":{"player":playerID,"leagueType":"league","sort":"season"},"query": QUERY}
    with httpx.Client(http2=True, headers=HEADERS) as client:
        return client.post(URL, json=eliteProspectsPayloadData)

def fetchValidPlayerIDs():
    validIDs = []
    offset = 0
    done = False
    while not done:
        try:
            print(f"Fetching players with offset {offset}")
            eliteProspectsPayloadData = {"variables":{"sort": "name", "limit":100, "offset": offset}, "query": SecondQuery}
            with httpx.Client(http2=True, headers=HEADERS) as client:
                response = client.post(URL, json=eliteProspectsPayloadData)
                # print(response.text)
                if response.is_success:
                    # parse data
                    try:
                        playerData = json.loads(response.text)
                        validPlayers = playerData.get("data", {}).get("players", {}).get("edges", {})
                        for player in validPlayers:
                            validIDs.append(player.get("id"))
                    except (TypeError, AttributeError) as e:
                        print(f"Error parsing data: {e}")
                        done = True
                offset += 100
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
    return validIDs

def getPlayerData(file, playerID, tries=0):
    try:
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
                    if player.get("height", {}):
                        playerStats["height"] = player.get("height", {}).get("metrics", {}) #cm
                    if player.get("weight", {}):
                        playerStats["weight"] = player.get("weight", {}).get("metrics", {}) #kg
                else:
                    return
                
                if playerStats["position"] == 'G':
                    return

                # itterate though each season played and collect stats about it
                playerStats["seasonStats"] = []

                inNHL = True
                for season in seasons:
                    team = season.get("teamName")
                    league = season.get("leagueName")
                    if league == "NHL":
                        inNHL = True
                    status = season.get("status")
                    year = season.get("season", {}).get("slug")

                    # Potentially remove check for valid leagues
                    if season.get("regularStats") and team:
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
                if inNHL:
                    file.write(json.dumps(playerStats, ensure_ascii=False)+"\n")
            except (TypeError, AttributeError) as e:
                print(f"Error parsing data for player ID {playerID}: {e}")
        else:
            print(f'''Error with response\n
                    Status Code: {response.status_code}\n
                    Data: {response.text}''')
    except httpx.HTTPError as e:
        print(f"HTTP error occurred for player ID {playerID}: {e}")
        if tries < 5:
            getPlayerData(file, playerID, tries=tries+1)


def main():
    print("fetching valid player IDs")
    validIDs = fetchValidPlayerIDs()
    # print(f"Scraping player data from ID {sys.argv[1]} to {sys.argv[2]}")
    # with open(f"data\\playerDataIDs{sys.argv[1]}-{sys.argv[2]}.txt", mode="w",encoding="utf-8") as file:
    print("scraping player data")
    with open(f"playerData.txt", mode="x",encoding="utf-8") as file:
        # for playerID in range(max(int(sys.argv[1]),1),int(sys.argv[2]) + 1):
        for playerID in validIDs:
            getPlayerData(file, playerID)
            # if playerID % 100 == 0 : print(f"reached id {playerID}")
            # print(f"Scraping data for player ID {playerID}")
            # get data
            

if __name__ == '__main__':
    main()