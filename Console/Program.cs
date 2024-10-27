using System.Diagnostics;
using System.Text;
using Engine;
using Engine.GameApiClient;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using OpenAI;

var session = "Bdet9d890000";
var gameClient = new ZorkApiClient();
var chatClient = new ChatGPTClient(CreateLogger());
var history = new LimitedStack<(string, string)>(15);
var map = new LimitedStack<string>(25);
var memory = new Memory(35);
string lastLocation = "West Of House";

chatClient.SystemPrompt = """
                          You are playing Zork One with the goal of winning the game by achieving a score of 350 points. Play as if for the first time, without relying on any prior knowledge of the Zork games.

                          Objective: Reach a score of 350 points.
                          Input Style: Use simple commands with one verb and one or two nouns, such as “OPEN DOOR” or “TURN SCREW WITH SCREWDRIVER.”
                          Inventory and Score: Type "INVENTORY" to check items you’re carrying and "SCORE" to view your current score.
                          Progression: Use the recent interactions with the game provided to avoid repeating actions you’ve already completed. Focus on logical actions that progress the game and explore new opportunities or areas based on the current context and past interactions.
                          """;

var lastResponse = await gameClient.GetAsync(new ZorkApiRequest("look", session));

if (lastResponse == null)
    throw new Exception("Null from Zork");

for (var i = 0; i < 150; i++)
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.WriteLine(lastResponse.Response);

    var historyString = new StringBuilder();
    var memoryString = new StringBuilder();
    var mapString = new StringBuilder();
    
    var counter = 1;
    var reverseHistory = history.GetAll().ToList();
    reverseHistory.Reverse();

    foreach (var item in memory.GetAll())
    {
        memoryString.AppendLine($"{counter}) {item.Remember}");
        counter++;
    }

    counter = 1;
    
    foreach (var line in reverseHistory)
    {
        historyString.AppendLine($"{counter}) The game said: {line.Item1} and you replied: {line.Item2}");
        counter++;
    }

    foreach (var line in map.GetAll())
    {
        mapString.AppendLine(line);
    }

    var request = new Request
    {
        UserMessage = $$"""
                        You have played {{lastResponse.Moves}} moves and have a score of {{lastResponse.Score}}. 
                        
                        You have navigated to and from the following locations:
                        
                        {{mapString}}
                        
                        You asked to be reminded of the following: 
                        
                        {{memoryString}}

                        Here are your recent interactions with the game, from most recent to least recent:

                        {{historyString}}

                        Most recently, the game has said this: "{{lastResponse.Response}}"
                            
                        Instructions: Based on your recent game history, memories and current context, provide a JSON output without backticks. Use the following format:

                        {   "command": "your command here. Choose commands that take logical next steps toward game progression and avoid previously attempted actions unless new clues or tools suggest a different outcome. Use your history to avoid going in circles. When stuck, explore new options. You want to go somewhere, you have to navigate manually using cardinal directions like NORTH, SOUTH, etc. Try all directions even if not listed as as possible exit", 
                            "reason": "brief explanation of why this command was chosen based on game state and history",
                            "remember": "Use this field only for unique, critical items, unsolved puzzles, or obstacles essential to game progress. These are like Leonard's tattoos in Memento. Memory is limited, so avoid duplicates or minor details. Leave empty if unnecessary. Do not repeat or duplicate reminders that already appear in the above prompt - these will still be remembered for you",
                            "rememberImportance": "the number, between 1 and 100, of how important the above reminder is, 1 is not really, 100 is critical. Lower number items are likely to be forgotten when we run out of memory."
                            "item": "any new items I have found, and their locations, which are not already mentioned above" 
                        }
                        """
    };

    GameResponse? chatResponse = JsonConvert.DeserializeObject<GameResponse>(await chatClient.CompleteChat(request));

    if (chatResponse is null)
        throw new Exception("Null from chat");
    
    if (chatResponse.Command is null)
        throw new Exception("Null command from chat");

    Thread.Sleep(TimeSpan.FromSeconds(5));
    
    Console.ForegroundColor = ConsoleColor.DarkGreen;
    Console.WriteLine("> " + chatResponse.Command);

    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine(chatResponse.Reason);
    
    if (!string.IsNullOrEmpty(chatResponse.Item))
    {
        //memory.Push(chatResponse);
        
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(chatResponse.Item);
    }
    
    if (chatResponse.RememberImportance > 0)
    {
        memory.Push(chatResponse);
        
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"{chatResponse.RememberImportance}: {chatResponse.Remember}");
    }

    lastResponse = await gameClient.GetAsync(new ZorkApiRequest(chatResponse.Command, session));
    if (lastResponse == null)
        throw new Exception("Null from Zork");
    
    if (lastResponse.LocationName != lastLocation)
    {
        string locationReminder =
            $"From: {lastLocation} To: {lastResponse.LocationName} Direction: {chatResponse.Command}";
        map.Push(locationReminder);
        lastLocation = lastResponse.LocationName;
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine(locationReminder);
    }
    
    history.Push((lastResponse.Response, chatResponse.Command));
}


ILogger CreateLogger()

{
    ILoggerFactory loggerFactory;

    if (Debugger.IsAttached)
        loggerFactory = LoggerFactory.Create(builder =>
            builder
                .AddConsole()
                .AddDebug()
                .AddFilter((category, _) =>
                {
                    if (category!.Contains("GameEngine.GameEngine"))
                        return true;

                    return false;
                })
                .SetMinimumLevel(LogLevel.Debug)
        );
    else
        loggerFactory = LoggerFactory.Create(builder =>
            builder
                .AddDebug()
                .AddFilter((category, _) =>
                {
                    if (category!.Contains("GameEngine.GameEngine"))
                        return true;

                    return false;
                })
                .SetMinimumLevel(LogLevel.Warning)
        );

    var logger = loggerFactory.CreateLogger<ChatGPTClient>();
    return logger;
}