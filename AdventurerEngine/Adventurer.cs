using Engine.GameApiClient;
using Newtonsoft.Json;
using OpenAI;

namespace AdventurerEngine;

public class Adventurer
{
    private readonly LimitedStack<(string, string)> _history = new(15);
    private readonly UniqueLimitedStack<string> _items = new(25);
    private readonly UniqueLimitedStack<string> _map = new(25);
    private readonly Memory _memory = new(35);
    private string _lastLocation = "West Of House";

    public string ItemString => Builders.BuildItems(_items);
    public string MapString => Builders.BuildMap(_map);
    public string HistoryString => Builders.BuildHistory(_history);
    public string MemoryString => Builders.BuildMemory(_memory);

    public async Task<AdventurerResponse> ChatResponse(ZorkApiResponse zorkApiResponse, ChatGPTClient chatGPTClient)
    {
        var request = new Request
        {
            UserMessage = $$"""
                            You have played {{zorkApiResponse.Moves}} moves and have a score of {{zorkApiResponse.Score}}. 

                            You are currently in this location: {{zorkApiResponse.LocationName}}

                            You have found the following items in the following locations:

                            {{ItemString}}

                            You have navigated to and from the following locations using these directions. Use this to build yourself a mental map:

                            {{MapString}}

                            You wanted to be reminded of the following important clues, problems and strategies: 

                            {{MemoryString}}

                            Here are your recent interactions with the game, from most recent to least recent. Study these carefully to avoid repeating yourself and going in circles: 

                            {{HistoryString}}
                                
                            Instructions: Based on your recent game history, memories and current context, provide a JSON output without backticks. Use the following format:

                            {   "command": "your command here. Choose commands that take logical next steps toward game progression and avoid previously attempted actions unless new clues or tools suggest a different outcome. Use your history to avoid going in circles. When stuck, explore new options. You want to go somewhere, you have to navigate manually using cardinal directions like NORTH, SOUTH, etc. Try all directions even if not listed as as possible exit", 
                                "reason": "brief explanation of why this command was chosen based on game state and history",
                                "remember": "Use this field only for new, novel or critical ideas for solving the game, new unsolved puzzles, or new obstacles essential to game progress. These are like Leonard's tattoos in Memento. Memory is limited, so avoid duplicates or minor details. Leave empty if unnecessary. Do not repeat yourself or duplicate reminders that already appear in the above prompt.",
                                "rememberImportance": "the number, between 1 and 1000, of how important the above reminder is, 1 is not really, 1000 is critical to winning the game. Lower number items are likely to be forgotten when we run out of memory."
                                "item": "any new, interesting items I have found in this location, along with their locations, which are not already mentioned above. For example "there is a box and a light bulb in the maintenance room". Omit if there is nothing here" 
                            }
                            """
        };

        var gameResponse = JsonConvert.DeserializeObject<AdventurerResponse>(await chatGPTClient.CompleteChat(request));

        if (gameResponse is null)
            throw new Exception("Null from chat");

        if (gameResponse.Command is null)
            throw new Exception("Null command from chat");

        _history.Push((zorkApiResponse.Response, gameResponse.Command));

        if (!string.IsNullOrEmpty(gameResponse.Item) && !gameResponse.Item.Contains("none "))
            _items.Push(gameResponse.Item);

        if (gameResponse.RememberImportance > 0) 
            _memory.Push(gameResponse);

        if (zorkApiResponse.LocationName != _lastLocation)
        {
            var locationReminder =
                $"From: {_lastLocation} To: {zorkApiResponse.LocationName} Direction: {gameResponse.Command}";
            _map.Push(locationReminder);

            _lastLocation = zorkApiResponse.LocationName;
        }
        
        _memory.Degrade();
        
        return gameResponse;
    }
}