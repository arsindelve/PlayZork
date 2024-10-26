using System.Diagnostics;
using System.Text;
using Engine;
using Engine.GameApiClient;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using OpenAI;

var session = "Budgeted8";
var gameClient = new ZorkApiClient();
var chatClient = new ChatGPTClient(CreateLogger());
var history = new LimitedStack<(string, string)>(15);
var memory = new Memory(35);

chatClient.SystemPrompt = """
                          You are playing Zork One with the goal of winning the game by achieving a score of 350 points. Play as if for the first time, without relying on any prior knowledge of the Zork games.

                          Objective: Reach a score of 350 points.
                          Input Style: Use simple commands with one verb and one or two nouns, such as “OPEN DOOR” or “TURN SCREW WITH SCREWDRIVER.”
                          Inventory and Score: Type "INVENTORY" to check items you’re carrying and "SCORE" to view your current score.
                          Progression: Use the history provided to avoid repeating actions you’ve already completed. Focus on logical actions that progress the game and explore new opportunities or areas based on the current context and past interactions.
                          """;

var lastResponse = await gameClient.GetAsync(new ZorkApiRequest("look", session));

if (lastResponse == null)
    throw new Exception("Null from Zork");

for (var i = 0; i < 200; i++)
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.WriteLine(lastResponse.Response);

    var historyString = new StringBuilder();
    var memoryString = new StringBuilder();
    
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

    var request = new Request
    {
        UserMessage = $$"""
                        You have played {{lastResponse.Moves}} moves and have a score of {{lastResponse.Score}}. 
                        
                        You have asked to be reminded of the following: 
                        
                        {{memoryString}}

                        Here are the last outputs from the game and your previous replies, from most recent to least recent:

                        {{historyString}}

                        Most recently, the game has said this: "{{lastResponse.Response}}"
                            
                        Instructions: Based on the game history and current context, provide a JSON output without backticks. Use the following format:

                        {   "command": "your command here", 
                            "reason": "brief explanation of why this command was chosen based on game state and history",
                            "remember": "if there is anything you want to 'remember' for later such as an unsolved puzzle or important, put it here and I will remember it for you. Your memory is limited so use this sparingly, and leave this empty unless it's important",
                            "rememberImportance": "the number, between 1 and 100, of how important the above reminder is, 1 is not really, 100 is critical. Lower number items are likely to be forgotten when we run out of memory."
                        }
                        """
    };

    var chatResponse = JsonConvert.DeserializeObject<GameResponse>(await chatClient.CompleteChat(request));

    if (chatResponse is null)
        throw new Exception("Null from chat");
    
    if (chatResponse.Command is null)
        throw new Exception("Null command from chat");

    Thread.Sleep(TimeSpan.FromSeconds(5));
    
    Console.ForegroundColor = ConsoleColor.DarkGreen;
    Console.WriteLine("> " + chatResponse.Command);

    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine(chatResponse.Reason);

    if (chatResponse.RememberImportance > 0)
    {
        memory.Push(chatResponse);
        
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"{chatResponse.RememberImportance}: {chatResponse.Remember}");
    }

    lastResponse = await gameClient.GetAsync(new ZorkApiRequest(chatResponse.Command, session));

    if (lastResponse == null)
        throw new Exception("Null from Zork");

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