using System.Diagnostics;
using Engine.GameApiClient;
using Microsoft.Extensions.Logging;
using OpenAI;

var session = "Matlock";
var gameClient = new ZorkApiClient();
var chatClient = new ChatGPTClient(CreateLogger());

var lastResponse = await gameClient.GetAsync(new ZorkApiRequest("look", session));

for (var i = 0; i < 10; i++)
{
    
    Console.ForegroundColor= ConsoleColor.White;
    Console.WriteLine(lastResponse.Response);
    
    if (lastResponse == null)
        throw new Exception("Null from Zork");

    chatClient.SystemPrompt = $"""
                               You are playing Zork One to win. You will win when you have a score of 350 points. 
                               The game responds best to very simple sentences of one verb and one or two nouns
                               such as "OPEN DOOR" or "TURN THE SCREW WITH THE SCREWDRIVER"

                               You have played {lastResponse.Moves} moves and have a score of {lastResponse.Score}. 
                               Your last response from the game was: {lastResponse.Response}

                               """;

    var request = new Request
        { UserMessage = " What is the next input you want to give to the game to progress toward a win. Provide only the input without explanation or commentary, as it is going to be passed straight to the game" };

    var chatResponse = await chatClient.CompleteChat(request);

    Console.ForegroundColor= ConsoleColor.DarkGreen;
    Console.WriteLine("> " + chatResponse);
    
    lastResponse = await gameClient.GetAsync(new ZorkApiRequest(chatResponse, session));
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